"""Measure whether the proxy actually saves money.

The premise of this project is a claim about cost, so it ships with the means
to check that claim against a real upstream, on a corpus that deliberately
includes workloads where translation is expected to *lose*: three-word prompts
whose translation fee dwarfs the request, and code-heavy payloads that barely
compress.

Savings are reported in dollars rather than tokens, because the translator
runs on a cheap model and the request on an expensive one. A verdict of
``loses`` is a valid, expected result for some categories.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

CORPUS_PATH = pathlib.Path(__file__).with_name("bench_corpus.jsonl")

#: A category has to clear this to count as worth using.
PAYS_OFF = 0.25


def load_corpus(path: pathlib.Path | None = None) -> list[dict]:
    """Read the bundled prompt corpus."""
    target = path or CORPUS_PATH
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def verdict_for(ratio: float) -> str:
    if ratio >= PAYS_OFF:
        return "pays off"
    if ratio > 0:
        return "marginal"
    return "loses"


@dataclass
class BenchResult:
    """One prompt's measurement."""

    id: str
    lang: str
    category: str
    baseline_cost: float = 0.0
    proxied_cost: float = 0.0
    translator_cost: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    #: Whether the proxy reported translating this request. A translated
    #: request with a zero fee means the fee was not measured, which would
    #: overstate the saving.
    translated: bool = False
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def dollars_saved(self) -> float:
        return self.baseline_cost - self.proxied_cost


def _bucket(results: list[BenchResult]) -> dict:
    baseline = sum(r.baseline_cost for r in results)
    proxied = sum(r.proxied_cost for r in results)
    translator = sum(r.translator_cost for r in results)
    ratio = (baseline - proxied) / baseline if baseline > 0 else 0.0
    return {
        "n": len(results),
        "baseline_cost": round(baseline, 6),
        "proxied_cost": round(proxied, 6),
        "translator_cost": round(translator, 6),
        "dollars_saved": round(baseline - proxied, 6),
        "savings_ratio": round(ratio, 4),
        "verdict": verdict_for(ratio) if results else "no data",
    }


def build_report(results: list[BenchResult]) -> dict:
    """Aggregate per-category and overall figures."""
    good = [r for r in results if r.ok]

    categories: dict[str, list[BenchResult]] = {}
    for result in good:
        categories.setdefault(result.category, []).append(result)

    return {
        "overall": _bucket(good),
        "categories": {name: _bucket(rows) for name, rows in sorted(categories.items())},
        "errors": sum(1 for r in results if not r.ok),
        # A truncated reply pins both runs to the same output length and hides
        # real savings, so the count is surfaced rather than buried.
        "truncated": sum(1 for r in good if r.detail.get("truncated")),
        # A request the proxy translated must carry a fee. If it does not, the
        # fee went unmeasured and every saving below it is overstated -- the
        # one failure mode that makes this tool lie in its own favour.
        "unmeasured_fee": sum(1 for r in good if r.translated and r.translator_cost <= 0.0),
    }


def exit_code_for(report: dict, *, min_savings: float) -> int:
    """Non-zero when the headline claim does not hold on real traffic."""
    # An unmeasured fee is worse than a bad result: the numbers are wrong in
    # this tool's own favour, so the run must not be read as a pass.
    if report.get("unmeasured_fee"):
        return 2

    chat = report["categories"].get("chat")
    if not chat or chat["n"] == 0:
        return 0
    return 0 if chat["savings_ratio"] >= min_savings else 1


def render(report: dict, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="lingua-proxy savings benchmark")
    table.add_column("Category")
    table.add_column("n", justify="right")
    table.add_column("Without proxy", justify="right")
    table.add_column("With proxy", justify="right")
    table.add_column("Saved", justify="right")
    table.add_column("Verdict")

    for name, bucket in report["categories"].items():
        colour = {"pays off": "green", "marginal": "yellow", "loses": "red"}.get(
            bucket["verdict"], "white"
        )
        table.add_row(
            name,
            str(bucket["n"]),
            f"${bucket['baseline_cost']:.5f}",
            f"${bucket['proxied_cost']:.5f}",
            f"{bucket['savings_ratio'] * 100:.1f}%",
            f"[{colour}]{bucket['verdict']}[/{colour}]",
        )

    overall = report["overall"]
    table.add_section()
    table.add_row(
        "[bold]overall[/bold]",
        str(overall["n"]),
        f"${overall['baseline_cost']:.5f}",
        f"${overall['proxied_cost']:.5f}",
        f"{overall['savings_ratio'] * 100:.1f}%",
        overall["verdict"],
    )
    console.print(table)

    if report["errors"]:
        console.print(f"[yellow]{report['errors']} prompt(s) failed and were excluded.[/yellow]")

    if report.get("truncated"):
        console.print(
            f"[yellow]{report['truncated']} reply/replies hit the token ceiling.[/yellow] "
            "Truncated replies understate savings, because both runs are forced to the "
            "same output length. Raise max_tokens for a cleaner measurement."
        )

    if report.get("unmeasured_fee"):
        console.print(
            f"[red]{report['unmeasured_fee']} translated request(s) recorded a translator "
            "fee of $0.00.[/red] That is not a free translation: the fee went unmeasured, "
            "so every saving above is overstated. Treat this run as invalid rather than "
            "as good news, and check that the proxy is writing its cost log."
        )


def run_bench(*, mode: str = "estimate", min_savings: float = 0.30, as_json: bool = False) -> int:
    """Run the benchmark against a configured upstream.

    Requires an explicit upstream: silently defaulting to the public API would
    spend the user's money without asking.
    """
    import os

    console = Console()
    upstream = os.environ.get("LINGUA_BENCH_BASE_URL")
    if not upstream:
        console.print(
            "[red]No benchmark upstream configured.[/red]\n"
            "Set LINGUA_BENCH_BASE_URL (and LINGUA_BENCH_AUTH if it needs a credential).\n"
            "This command sends real requests and spends real money, so it will not "
            "guess an endpoint for you."
        )
        return 2

    from lingua_proxy.bench_runner import run_live_bench

    report = run_live_bench(upstream=upstream, mode=mode)
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        render(report, console)
    return exit_code_for(report, min_savings=min_savings)
