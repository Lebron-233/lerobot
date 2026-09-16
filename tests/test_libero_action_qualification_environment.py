"""CPU-only regression for the F-ACQ1-R1 launch fix; no real native imports."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_action_qualification as q  # noqa: E402


@pytest.fixture
def environment(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("synthetic fixed config\n")
    assets = tmp_path / "assets"
    assets.mkdir()
    library = tmp_path / "libGL.so"
    library.write_text("synthetic library\n")
    values = {**q.RUNTIME_ENV, "LIBERO_CONFIG_PATH": str(tmp_path), "LD_PRELOAD": str(library)}
    monkeypatch.setattr(q, "RUNTIME_ENV", values)
    monkeypatch.setattr(q, "CONFIG_SHA256", q.digest(config))
    monkeypatch.setattr(q, "ASSETS_LINK", assets)
    monkeypatch.setattr(q, "ASSETS_PATH", assets)
    monkeypatch.setattr(q.e, "PYTHON", sys.executable)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return config, assets


def test_environment_is_read_only(environment):
    config, _ = environment
    before = config.read_bytes()
    assert q.runtime_environment()["cuda_initialized"] is False
    assert config.read_bytes() == before


@pytest.mark.parametrize("key", sorted(q.RUNTIME_ENV))
def test_missing_variable_rejected_before_import(environment, monkeypatch, key):
    monkeypatch.delenv(key)
    with pytest.raises(ValueError, match="launch environment"):
        q.runtime_environment()


def test_inherited_pythonpath_rejected(environment, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    with pytest.raises(ValueError, match="PYTHONPATH_present=True"):
        q.runtime_environment()


def test_missing_config_not_created(environment):
    config, _ = environment
    config.unlink()
    with pytest.raises(ValueError, match="configuration missing or changed"):
        q.runtime_environment()
    assert not config.exists()


def test_config_and_asset_tampering_rejected(environment, monkeypatch):
    config, assets = environment
    config.write_text("changed")
    with pytest.raises(ValueError, match="configuration missing or changed"):
        q.runtime_environment()
    monkeypatch.setattr(q, "CONFIG_SHA256", q.digest(config))
    monkeypatch.setattr(q, "ASSETS_PATH", assets / "wrong")
    with pytest.raises(ValueError, match="assets link differs"):
        q.runtime_environment()


def test_worker_preflight_failure_does_not_load_models(tmp_path, monkeypatch):
    monkeypatch.setattr(q.acr, "source_gate", lambda _: None)
    monkeypatch.setattr(q, "validate_preparation", lambda _: None)

    def reject():
        raise ValueError("synthetic environment rejection")

    def forbidden(*args, **kwargs):
        pytest.fail("Factory or model loading reached after failed preflight")

    monkeypatch.setattr(q, "runtime_environment", reject)
    monkeypatch.setattr(q.e.reference, "make_native_env_factory", forbidden)
    monkeypatch.setattr(q.e, "load_runtime", forbidden)
    assert q.worker(SimpleNamespace(execution_head="a" * 40, output=tmp_path)) == 2
    result = json.loads((tmp_path / "worker_result.json").read_text())
    assert "synthetic environment rejection" in result["first_failure"]
    assert result["counts"] == {} and result["native_budget"] == {}
    assert not q.torch.cuda.is_initialized()
