"""Shared test fixtures."""

import numpy as np
import pandas as pd

from driftalytics import Monitor, MonitorResult


def data(n=500, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(loc=shift, size=n),
            "b": rng.normal(size=n),
        }
    )


def write_damage_csv(path, rows=30, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "drift_magnitude": rng.uniform(0, 1, rows),
            "missingness_delta": rng.uniform(0, 0.2, rows),
            "cardinality_shift": rng.uniform(0, 0.5, rows),
            "schema_change": rng.uniform(0, 0.2, rows),
            "volume_delta": rng.uniform(0, 0.2, rows),
        }
    )
    frame["damage"] = np.clip(
        0.3 * frame["drift_magnitude"]
        + 0.2 * frame["missingness_delta"]
        + 0.1 * frame["cardinality_shift"],
        0,
        1,
    )
    frame.to_csv(path, index=False)


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.requests = []

    def complete(self, prompt, system=None, max_output_tokens=256):
        self.calls += 1
        self.requests.append(
            {
                "prompt": prompt,
                "system": system,
                "max_output_tokens": max_output_tokens,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response, 10


class AlwaysAlert(Monitor):
    name = "always"

    def on_batch(self, reference, batch):
        return MonitorResult(name=self.name, score=0.5, alert=False)
