"""Record what the proxy rewrote, so a bad translation can be diagnosed.

No automated check can catch a translation that is fluent, structurally valid
and simply wrong -- "do not delete this" rendered as "delete this" passes every
test in this codebase. Since the failure cannot be detected, it must at least be
*visible*: when an answer looks wrong, the user needs to see the English the
model actually received to tell a misunderstanding from a mistranslation.

Off by default, because unlike the cost log this necessarily contains prompt
text.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import time


class AuditLog:
    """Append-only record of translated prompt/reply pairs."""

    def __init__(self, path: pathlib.Path | None):
        self.path = pathlib.Path(path) if path else None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(
        self,
        *,
        direction: str,
        source_lang: str,
        target_lang: str,
        original: str,
        translated: str,
        model: str = "",
    ) -> None:
        if self.path is None:
            return

        entry = {
            "v": 1,
            "ts": int(time.time()),
            "direction": direction,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original": original,
            "translated": translated,
            "model": model,
        }

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existed = self.path.exists()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            if not existed:
                # Contains prompt text; keep it owner-only.
                with contextlib.suppress(OSError):
                    os.chmod(self.path, 0o600)
        except OSError:
            # Auditing must never break the request it is observing.
            pass

    def entries(self, limit: int | None = None) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out[-limit:] if limit else out
