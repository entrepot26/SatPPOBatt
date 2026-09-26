from __future__ import annotations


class SimulationClock:
    def __init__(self) -> None:
        self._time = 0.0

    @property
    def now(self) -> float:
        return self._time

    def reset(self, value: float = 0.0) -> None:
        self._time = float(value)

    def advance(self, delta: float) -> float:
        self._time += float(delta)
        return self._time

