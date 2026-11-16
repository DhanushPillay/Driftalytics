"""Damage predictor and training tests."""

import json

import numpy as np
import pytest
from helpers import write_damage_csv

from driftalytics import CorruptionVector, DamagePredictor
from driftalytics.train_damage import train_and_save


def test_damage_predictor_fit_save_load(tmp_path):
    features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    target = np.array([0.0, 0.3, 0.25, 0.5])
    model = DamagePredictor().fit(features, target)
    prediction = model.predict(CorruptionVector(1.0, 0.0, 0.0, 0.0, 0.0))
    assert 0.0 <= prediction <= 1.0

    path = tmp_path / "damage.json"
    model.save(str(path))
    loaded = DamagePredictor.load(str(path))
    assert loaded.predict(CorruptionVector(1.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx(prediction)
    with pytest.raises(ValueError):
        loaded.predict([1.0])


def test_train_damage_writes_json(tmp_path):
    data = tmp_path / "damage.csv"
    write_damage_csv(data)
    path = tmp_path / "model.json"
    train_and_save(str(path), str(data))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "driftalytics-damage-v1"
    assert len(payload["coef"]) == 5


def test_damage_metrics_and_invalid_model_payload(tmp_path):
    features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    target = np.array([0.0, 0.3, 0.25, 0.5])
    model = DamagePredictor().fit(features, target)
    model.validation_metrics_ = model.evaluate(features[:2], target[:2])
    model.test_metrics_ = model.evaluate(features[2:], target[2:])
    path = tmp_path / "damage.json"
    model.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["training_metrics"] == payload["metrics"]
    assert "validation_metrics" in payload and "test_metrics" in payload

    payload["coef"][0] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        DamagePredictor.load(str(path))


def test_train_damage_saves_validation_and_test_metrics(tmp_path):
    data = tmp_path / "damage.csv"
    write_damage_csv(data)
    path = tmp_path / "model.json"
    train_and_save(str(path), str(data))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"] == payload["training_metrics"]
    assert payload["validation_metrics"]
    assert payload["test_metrics"]
    with pytest.raises(ValueError):
        train_and_save(str(path), str(data), test_size=0.6)


def test_train_damage_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_and_save(str(tmp_path / "model.json"), str(tmp_path / "missing.csv"))
