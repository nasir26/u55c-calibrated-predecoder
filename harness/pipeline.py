"""Decoding pipeline on calibrated circuits: detectors -> NVIDIA model input -> pre-decoder (float, integer model or
FPGA bits) -> residual detectors -> PyMatching, through NVIDIA's evaluation module (Ising-Decoding, Apache-2.0).

Author: Nasir Ali, C-DAC Noida.

The pre-decoder's corrections, the residual syndrome and the logical frame are computed by
`evaluation.logical_error_rate.PreDecoderMemoryEvalModule`, unchanged; only the circuit (calib/noise.py) and the
matching graph (from the calibrated, the median or NVIDIA's nominal circuit) are ours. The final prediction is
pre_L xor PyMatching(residual), as in NVIDIA's `run_inference_and_decode_pre_decoder_memory`.

torch.compile is disabled (TORCHDYNAMO_DISABLE=1): the host compiler is too old for it; eager execution computes the
same function.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
REF = Path(os.environ.get("ISING_DECODING_DIR", ROOT / "third_party" / "Ising-Decoding")).resolve()
MODELS = Path(os.environ.get("NPD_MODELS", ROOT / "runs" / "models")).resolve()
for p in (str(REF / "code"), str(ROOT), str(ROOT / "model"), str(ROOT / "host")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pymatching  # noqa: E402
import stim  # noqa: E402
import torch  # noqa: E402

FAST = "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"


def make_cfg(d: int, rounds: int, basis: str, model_id: int = 1):
    """The configuration NVIDIA's run.py builds for public surface code inference, with the test basis set."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(REF / "conf")):
        cfg = compose(config_name="config_public", overrides=[f"distance={d}", f"n_rounds={rounds}",
                                                               f"model_id={model_id}", "workflow.task=inference"])
    cfg = apply_public_defaults_and_model(cfg, validate_public_config(cfg))
    with open_dict(cfg):
        cfg.test.meas_basis_test = basis
        cfg.test.n_rounds = rounds
    return cfg


def load_fast(path: Path | None = None) -> torch.nn.Module:
    """NVIDIA's Ising-Decoder-SurfaceCode-1-Fast checkpoint (NVIDIA Open Model License) in float32."""
    from export.safetensors_utils import load_safetensors
    model, _ = load_safetensors(str(path or MODELS / "nvidia" / FAST), device="cpu")
    return model.float().eval()


class Fixed(torch.nn.Module):
    """Stand in network that returns precomputed logits (for decisions made elsewhere, e.g. by the FPGA)."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out = None

    def forward(self, x):
        return self.out


class Pipeline:
    def __init__(self, d: int, rounds: int, basis: str, model: torch.nn.Module | None = None):
        from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
        self.d, self.rounds, self.basis = d, rounds, basis
        self.cfg = make_cfg(d, rounds, basis)
        self.fixed = Fixed()
        self.mod = PreDecoderMemoryEvalModule(model if model is not None else self.fixed, self.cfg,
                                              _build_stab_maps(d, "XV"), torch.device("cpu")).eval()

    def inputs(self, dets: np.ndarray) -> torch.Tensor:
        """trainX (B, 4, T, d, d) float, as NVIDIA builds it."""
        with torch.no_grad():
            return self.mod._batch_to_trainx_and_syndromes(torch.from_numpy(dets.astype(np.uint8)))[0]

    def run(self, dets: np.ndarray, logits: torch.Tensor | None = None, batch: int = 4096):
        """pre_L (B,) and residual detectors (B, n_det) for the module's network, or for given logits."""
        pre, res = [], []
        with torch.no_grad():
            for s in range(0, len(dets), batch):
                x = torch.from_numpy(dets[s:s + batch].astype(np.uint8))
                if logits is not None:
                    self.fixed.out = logits[s:s + batch]
                    self.mod.model = self.fixed
                o = self.mod(x).numpy()
                pre.append(o[:, 0])
                res.append(o[:, 1:])
        return np.concatenate(pre), np.concatenate(res)


def matcher(circuit: stim.Circuit) -> pymatching.Matching:
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def fails(m: pymatching.Matching, dets: np.ndarray, obs: np.ndarray, pre_l: np.ndarray | None = None) -> np.ndarray:
    """Per shot failure bits of PyMatching on dets, with the pre-decoder's logical frame folded in if given."""
    pred = m.decode_batch(dets)[:, 0].astype(np.uint8)
    if pre_l is not None:
        pred ^= pre_l.astype(np.uint8)
    return pred != obs[:, 0]


def bits_to_logits(bits: np.ndarray) -> torch.Tensor:
    """Decision bits (B, 4, T, d, d) to logits that NVIDIA's threshold (0) turns back into the same bits."""
    return torch.from_numpy(np.where(bits.astype(bool), 1.0, -1.0).astype(np.float32))
