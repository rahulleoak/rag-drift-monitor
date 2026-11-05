"""Named time periods (baseline and later periods) from the config."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class Period:
    """A named, inclusive date range."""

    name: str
    start: date
    end: date

    @classmethod
    def from_config(cls, d: Mapping[str, Any]) -> Period:
        """Build from a config entry ``{name, start, end}`` with ISO dates."""
        return cls(str(d["name"]), date.fromisoformat(str(d["start"])),
                   date.fromisoformat(str(d["end"])))

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def halves(self) -> tuple[Period, Period]:
        """Split at the midpoint, e.g. 2016-17 into 2016 and 2017."""
        mid = self.start + (self.end - self.start) // 2
        return (Period(f"{self.name} (1st half)", self.start, mid),
                Period(f"{self.name} (2nd half)", mid + timedelta(days=1), self.end))


def study_periods(config: Mapping[str, Any]) -> list[Period]:
    """The baseline followed by the later periods, in config order.

    Raises ``ValueError`` if any two periods overlap, because every document
    must belong to at most one period.
    """
    periods = [Period.from_config(config["periods"]["baseline"])]
    periods += [Period.from_config(d) for d in config["periods"]["later"]]
    ordered = sorted(periods, key=lambda p: p.start)
    for a, b in zip(ordered, ordered[1:]):
        if b.start <= a.end:
            raise ValueError(f"periods {a.name!r} and {b.name!r} overlap")
    return periods


def assign_period(day: date, periods: Iterable[Period]) -> str | None:
    """Name of the period containing ``day``, or None if it is in none."""
    for p in periods:
        if p.contains(day):
            return p.name
    return None
