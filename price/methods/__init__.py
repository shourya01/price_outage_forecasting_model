"""Registry of the methods in the main results table."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Callable


@dataclasses.dataclass(frozen=True)
class Method:
    """One row of the main table: key, display label, implementing module in price.methods, fixed keyword
    arguments, whether it uses weather, and whether it runs on the GPU queue.
    """

    key: str
    label: str
    module: str
    kwargs: dict
    uses_weather: bool
    gpu: bool = False

    def run(self, exp, **kw) -> "np.ndarray":
        """Import the method's module and call its `run(exp, ...)`; `kw` overrides the fixed keyword arguments."""
        fn: Callable = importlib.import_module(f"price.methods.{self.module}").run
        return fn(exp, **{**self.kwargs, **kw})


METHODS = {
    m.key: m
    for m in [
        Method("zeros", "All-Zeros", "zeros", {}, uses_weather=False),
        Method("arima", "ARIMA(1,0,1)", "arima", {"exog": False}, uses_weather=False),
        Method("arimax", "ARIMAX(1,0,1)", "arima", {"exog": True}, uses_weather=True),
        Method("lstm", "LSTM", "lstm", {"sparsity_clusters": False}, uses_weather=True, gpu=True),
        Method(
            "lstm_sparsity", "LSTM (Sparsity Cl.)", "lstm", {"sparsity_clusters": True}, uses_weather=True, gpu=True
        ),
        Method("zip", "ZIP", "zip", {}, uses_weather=True),
        Method("tshm", "TSHM", "tshm", {}, uses_weather=True),
        Method("timesfm", "TimesFM 3.0 (zero-shot)", "timesfm", {}, uses_weather=True, gpu=True),
        Method("price", "PRiCE", "price", {}, uses_weather=True),
    ]
}
TABLE_ORDER = ["zeros", "arima", "arimax", "lstm", "lstm_sparsity", "zip", "tshm", "timesfm", "price"]
