"""The real inference package must remain usable without the dataset extra."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_python(source):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_inference_without_datasets_keeps_canonical_classes_and_missing_extra_error():
    run_python("""
import sys
from importlib.abc import MetaPathFinder
from lerobot.utils import import_utils

# Simulate only optional dependency discovery, never fake a rollout module.
import_utils._datasets_available = False
import_utils._require_package_cache['datasets'] = False
class RejectDatasets(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'datasets' or fullname.startswith(('datasets.', 'lerobot.datasets')):
            raise AssertionError('Standalone inference imported datasets: ' + fullname)
sys.meta_path.insert(0, RejectDatasets())

import lerobot.rollout as rollout
from lerobot.rollout.inference import (
    PredictiveAsyncInferenceEngine, RTCInferenceEngine, SyncInferenceEngine,
    create_inference_engine,
)
from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine as canonical
assert rollout.PredictiveAsyncInferenceEngine is canonical is PredictiveAsyncInferenceEngine
assert rollout.SyncInferenceEngine is SyncInferenceEngine
assert rollout.RTCInferenceEngine is RTCInferenceEngine
assert rollout.create_inference_engine is create_inference_engine
for name in ('RolloutContext', 'InteractiveSession', 'create_strategy'):
    try:
        getattr(rollout, name)
    except ImportError as error:
        assert "'datasets' is required but not installed" in str(error)
    else:
        raise AssertionError('Missing dataset extra was silently accepted: ' + name)
try:
    rollout.not_a_public_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute did not fail normally')
assert not any(n == 'datasets' or n.startswith('datasets.') for n in sys.modules)
""")


def test_full_rollout_exports_remain_available_with_dataset_extra():
    from lerobot.utils.import_utils import _datasets_available

    if not _datasets_available:
        pytest.skip("The full-rollout branch is checked in the existing dataset-enabled CPU environment")
    run_python("""
import lerobot.rollout as rollout
from lerobot.rollout.context import RolloutContext
from lerobot.rollout.controller import RolloutController
from lerobot.rollout.interactive import InteractiveSession
assert rollout.RolloutContext is RolloutContext
assert rollout.RolloutController is RolloutController
assert rollout.InteractiveSession is InteractiveSession
assert all(getattr(rollout, name) is not None for name in rollout.__all__)
""")


def test_real_d3_entry_imports_before_any_model_or_input_work():
    run_python("""
import runpy
import sys
from pathlib import Path
import torch
root = Path.cwd()
directory = root / 'examples/advanced/predictive_async'
sys.path.insert(0, str(directory))
namespace = runpy.run_path(str(directory / 'validate_smolvla_graph_worker.py'), run_name='d3_import_only')
from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine
assert issubclass(namespace['SmolVLAGraphIdentityEngine'], PredictiveAsyncInferenceEngine)
assert len(namespace['WORKER_EVENTS']) == 12
assert not torch.cuda.is_initialized()
assert 'main' in namespace  # Import only: main/loader/fixed_inputs are not called.
""")
