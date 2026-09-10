"""Command line interface."""

from __future__ import annotations

import json
import os
import subprocess

import click
import httpx
from rich.console import Console
from rich.table import Table

from lingua_proxy import __version__
from lingua_proxy.config import (
    Settings,
    default_config_path,
    lingua_home,
    resolve_anthropic_upstream,
    resolve_openai_upstream,
)
from lingua_proxy.cost_log import CostLog, summarize
from lingua_proxy.doctor import run_checks
from lingua_proxy.wrap import (
    WrapError,
    apply_wrap,
    clear_marker,
    find_claude_binary,
    load_marker,
    restore_wrap,
)

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="lingua-proxy")
def main() -> None:
    """Translate non-English LLM traffic to English to cut token costs."""


# -- proxy --------------------------------------------------------------


@main.command()
@click.option("--host", default=None, help="Bind address (loopback by default).")
@click.option("--port", type=int, default=None, help="Port to listen on.")
@click.option("--upstream", default=None, help="Anthropic-format upstream base URL.")
@click.option("--openai-upstream", default=None, help="OpenAI-format upstream base URL.")
@click.option("--translator-model", default=None, help="Model used for translation.")
def proxy(
    host: str | None,
    port: int | None,
    upstream: str | None,
    openai_upstream: str | None,
    translator_model: str | None,
) -> None:
    """Run the translating proxy."""
    import uvicorn

    from lingua_proxy.proxy import create_app

    settings = Settings(
        upstream_anthropic_url=resolve_anthropic_upstream(upstream),
        upstream_openai_url=resolve_openai_upstream(openai_upstream),
    )
    if host:
        settings.host = host
    if port:
        settings.port = port
    if translator_model:
        settings.translator_model = translator_model

    console.print(f"[bold]lingua-proxy[/bold] {__version__}")
    console.print(f"  listening on  http://{settings.host}:{settings.port}")
    console.print(f"  upstream      {settings.upstream_anthropic_url}")
    console.print(f"  translator    {settings.translator_model}")

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


# -- wrap ---------------------------------------------------------------


def _proxy_is_up(port: int) -> bool:
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
        return response.json().get("service") == "lingua-proxy"
    except Exception:
        return False


def _persist_upstream(url: str) -> None:
    """Remember the resolved upstream so proxy/doctor/bench agree."""
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    lines = [ln for ln in existing.splitlines() if not ln.startswith("upstream_anthropic_url")]
    lines.append(f'upstream_anthropic_url = "{url}"')
    path.write_text("\n".join(lines).strip() + "\n")


@main.group()
def wrap() -> None:
    """Run a client with its traffic routed through lingua-proxy."""


@wrap.command("claude")
@click.option("--port", type=int, default=8787, show_default=True)
@click.option("--upstream", default=None, help="Override the detected upstream.")
@click.option("--no-proxy", is_flag=True, help="Assume the proxy is already running.")
@click.option("--dry-run", is_flag=True, help="Show what would change, then restore.")
@click.option("--keep-patched", is_flag=True, hidden=True, help="Testing aid: skip restore.")
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def wrap_claude(
    port: int,
    upstream: str | None,
    no_proxy: bool,
    dry_run: bool,
    keep_patched: bool,
    claude_args: tuple[str, ...],
) -> None:
    """Launch Claude Code pointed at the proxy, then restore your settings."""
    if os.environ.get("LINGUA_PROXY_WRAPPED"):
        raise click.ClickException(
            "This session is already wrapped by lingua-proxy. Nesting would loop traffic."
        )

    resolved = resolve_anthropic_upstream(upstream)
    proxy_url = f"http://127.0.0.1:{port}"
    if resolved.rstrip("/") == proxy_url:
        raise click.ClickException(
            "The detected upstream is lingua-proxy itself, which would loop. "
            "Pass --upstream to name the real API or gateway."
        )

    binary = None
    if not dry_run and not keep_patched:
        try:
            binary = find_claude_binary()
        except WrapError as exc:
            raise click.ClickException(str(exc)) from exc

    _persist_upstream(resolved)

    try:
        state = apply_wrap(proxy_url, port)
    except WrapError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[bold]lingua-proxy[/bold] wrapping Claude Code")
    console.print(f"  proxy      {proxy_url}")
    console.print(f"  upstream   {resolved}")
    console.print(f"  patched    {state.settings_path}")

    if not no_proxy and not _proxy_is_up(port):
        console.print(
            f"  [yellow]note[/yellow]  no proxy answering on port {port}; "
            f"start one with 'lingua-proxy proxy --port {port}'"
        )

    if keep_patched:
        return

    if dry_run:
        restore_wrap(state)
        clear_marker()
        console.print("  [green]dry run[/green] settings restored")
        return

    try:
        # Set the base URL on the child too. The settings file is what Claude
        # Code actually reads, but leaving a stale shell variable pointing
        # elsewhere is confusing and would apply to anything else it launches.
        child_env = {
            **os.environ,
            "ANTHROPIC_BASE_URL": proxy_url,
            "LINGUA_PROXY_WRAPPED": "1",
        }
        result = subprocess.run([binary, *claude_args], env=child_env)
        code = result.returncode
    finally:
        restore_wrap(state)
        clear_marker()
        console.print("  settings restored")

    raise SystemExit(code)


@main.group()
def unwrap() -> None:
    """Undo a wrap that did not clean up after itself."""


@unwrap.command("claude")
@click.option("--no-stop-proxy", is_flag=True, help="Leave a running proxy alone.")
def unwrap_claude(no_stop_proxy: bool) -> None:
    """Restore client settings after a crashed wrap."""
    state = load_marker()
    if state is None:
        console.print("Nothing to undo: no lingua-proxy wrap is recorded for this directory.")
        return

    restore_wrap(state)
    clear_marker()
    console.print(f"Restored {state.settings_path}")


# -- doctor -------------------------------------------------------------


@main.command()
@click.option("--port", type=int, default=8787, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def doctor(port: int, as_json: bool) -> None:
    """Check that everything is wired up correctly."""
    checks = run_checks(port=port)
    worst = max((c.severity for c in checks), default=0)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "version": __version__,
                    "exit_code": worst,
                    "checks": [c.to_json() for c in checks],
                },
                indent=2,
            )
        )
        raise SystemExit(worst)

    table = Table(title=f"lingua-proxy {__version__}", show_lines=False)
    table.add_column("")
    table.add_column("Check")
    table.add_column("Result")
    for check in checks:
        table.add_row(check.glyph, check.name, check.summary)
    console.print(table)

    for check in checks:
        if check.hint and check.severity:
            console.print(f"[yellow]hint[/yellow] {check.name}: {check.hint}")

    raise SystemExit(worst)


# -- dashboard ----------------------------------------------------------


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def dashboard(as_json: bool) -> None:
    """Show measured token savings."""
    log = CostLog(path=lingua_home() / "cost_log.jsonl")
    summary = summarize(log.rows())

    if as_json:
        click.echo(json.dumps(summary.to_json(), indent=2))
        return

    if summary.requests == 0:
        console.print(
            "No requests recorded yet. Start the proxy, send some traffic, then check back."
        )
        return

    table = Table(title="lingua-proxy savings")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Requests", str(summary.requests))
    table.add_row("Translated", str(summary.translated))
    table.add_row("Passed through", str(summary.passthrough))
    table.add_row("Cost with proxy", f"${summary.dollars_proxied:.4f}")
    table.add_row("Cost without", f"${summary.dollars_baseline:.4f}")
    table.add_row("Saved", f"${summary.dollars_saved:.4f}")
    table.add_row("Savings", f"{summary.savings_ratio * 100:.1f}%")
    table.add_row("Translator cost", f"${summary.translator_dollars:.4f}")
    console.print(table)

    if summary.by_language:
        langs = Table(title="By language")
        langs.add_column("Language")
        langs.add_column("Requests", justify="right")
        for lang, count in sorted(summary.by_language.items(), key=lambda kv: -kv[1]):
            langs.add_row(lang, str(count))
        console.print(langs)


# -- advise -------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="The model your requests actually use.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def advise(model: str, as_json: bool) -> None:
    """Say whether translation can pay off for a given model, and with what.

    Both the saving and the translation fee scale with the length of the
    answer, so the answer's length cancels out: what matters is how much
    shorter the English reply is. That threshold is arithmetic, so it can be
    computed rather than discovered by spending money on a sweep.
    """
    from lingua_proxy.cost_log import PRICES, break_even_shrink, price_for

    main_price = price_for(model)

    # Only translators reachable through the same upstream are useful here.
    # A cheaper model behind a different vendor's API is not a drop-in.
    family = "claude-" if model.startswith("claude-") else ""
    names = [n for n in sorted(PRICES) if n.startswith(family)] or sorted(PRICES)

    candidates = []
    for name in names:
        threshold = break_even_shrink(main_price, price_for(name))
        candidates.append(
            {
                "translator": name,
                "required_shrink": round(threshold, 4),
                "viable": threshold < 1.0,
            }
        )

    viable = [c for c in candidates if c["viable"]]
    best = min(viable, key=lambda c: c["required_shrink"]) if viable else None

    if as_json:
        click.echo(
            json.dumps({"model": model, "recommended": best, "candidates": candidates}, indent=2)
        )
        raise SystemExit(0 if best else 1)

    if best is None:
        console.print(
            f"[red]Translation is not worth it for {model}.[/red]\n"
            "Every available translator costs as much per output token as the model "
            "itself, so you would pay to rewrite an answer you already paid for."
        )
        raise SystemExit(1)

    table = Table(title=f"Translator options for {model}")
    table.add_column("Translator")
    table.add_column("English reply must be shorter by", justify="right")
    table.add_column("")
    for c in candidates:
        if not c["viable"]:
            table.add_row(c["translator"], "never pays", "[dim]—[/dim]")
        else:
            mark = "[green]recommended[/green]" if c is best else ""
            table.add_row(c["translator"], f"{c['required_shrink'] * 100:.0f}%", mark)
    console.print(table)

    # Range observed across the two recorded head-to-head runs; see
    # tests/fixtures/head_to_head.json.
    console.print(
        f"\nUse [bold]{best['translator']}[/bold]. Measured shrink ranges from about "
        f"14% on a sprawling answer to 72% on a tightly-scoped one, and a reply that "
        f"hits max_tokens cannot shrink at all. A {best['required_shrink'] * 100:.0f}% "
        "threshold is therefore cleared by focused questions and missed by open-ended "
        "ones. Run 'lingua-proxy bench' to measure your own traffic."
    )


# -- bench --------------------------------------------------------------


@main.command()
@click.option("--mode", type=click.Choice(["estimate", "ab"]), default="estimate")
@click.option("--min-savings", type=float, default=0.30, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def bench(mode: str, min_savings: float, as_json: bool) -> None:
    """Measure real savings against your own upstream."""
    from lingua_proxy.bench import run_bench

    raise SystemExit(run_bench(mode=mode, min_savings=min_savings, as_json=as_json))
