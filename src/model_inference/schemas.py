from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationSummary:
    total_rows: int
    generated_ok: int
    generated_errors: int


@dataclass(frozen=True)
class GenerationReport:
    schema_version: str
    config: dict[str, Any]
    summary: GenerationSummary
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config,
            "summary": asdict(self.summary),
            "rows": self.rows,
        }
