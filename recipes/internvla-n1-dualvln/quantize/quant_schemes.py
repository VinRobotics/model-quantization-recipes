# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Quantization scheme registry, validity gate, and ModelOpt config composer.

Schemes and strategies live in ``configs/schemes.yaml`` rather than in Python so the
matrix is readable without tracing code, following the ``qwen36-27b`` recipe's pattern.

The validity gate exists because two combinations are impossible on this hardware and
neither fails early on its own:

* NVFP4 cannot quantize the Qwen2.5-VL vision tower. The ViT MLP ``intermediate_size``
  is 3420 and the NVFP4 block size is 16; 3420 / 16 = 213.75. Without a gate this only
  surfaces once quantization is already under way.
* NVFP4 KV cache requires ``sm100f`` (datacenter Blackwell). Jetson Thor is sm110, so the
  KV cache is FP8 even when the weights are NVFP4.

The divisibility check reads ``vision_config.intermediate_size`` from the checkpoint being
quantized rather than hardcoding 3420, so it stays correct for other Qwen2.5-VL sizes.
"""
import copy
import json
import os
from typing import Any, Optional

import yaml

_DEFAULT_SCHEMES_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "schemes.yaml"
)

# NVFP4 packs weights in blocks of this many elements along the reduction axis.
NVFP4_BLOCK_SIZE = 16


def load_registry(path: Optional[str] = None) -> dict:
    """Load ``configs/schemes.yaml``."""
    path = path or _DEFAULT_SCHEMES_YAML
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scheme registry not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def scheme_names(registry: Optional[dict] = None) -> list[str]:
    return sorted((registry or load_registry())["schemes"])


def strategy_names(registry: Optional[dict] = None) -> list[str]:
    return sorted((registry or load_registry())["strategies"])


def is_nvfp4(scheme: str) -> bool:
    return scheme.startswith("nvfp4")


def _vision_intermediate_size(model_path: str) -> Optional[int]:
    """Read ``vision_config.intermediate_size`` from a checkpoint, if it has one."""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    vision = cfg.get("vision_config")
    if isinstance(vision, dict):
        return vision.get("intermediate_size")
    return None


def validate(scheme: str, strategy: str, model_path: Optional[str] = None,
             allow_experimental: bool = False,
             registry: Optional[dict] = None) -> None:
    """Raise ``ValueError`` if this combination cannot or should not run.

    Called before the model is loaded, so a rejected combination costs seconds rather
    than a full checkpoint load followed by a mid-quantization crash.
    """
    registry = registry or load_registry()

    if scheme not in registry["schemes"]:
        raise ValueError(f"unknown scheme {scheme!r}; available: {scheme_names(registry)}")
    if strategy not in registry["strategies"]:
        raise ValueError(f"unknown strategy {strategy!r}; available: {strategy_names(registry)}")

    strat = registry["strategies"][strategy]

    for rule in registry.get("blocked", []):
        if scheme in rule["schemes"] and strategy in rule["strategies"]:
            detail = ""
            # Prefer the checkpoint's own number over the one in the message.
            if strat["quantize_visual"] and model_path:
                size = _vision_intermediate_size(model_path)
                if size is not None:
                    detail = (f" This checkpoint's vision_config.intermediate_size is {size}; "
                              f"{size} / {NVFP4_BLOCK_SIZE} = {size / NVFP4_BLOCK_SIZE}.")
            raise ValueError(f"{scheme} + {strategy} is not supported. "
                             f"{rule['reason'].strip()}{detail}")

    for rule in registry.get("experimental", []):
        if scheme in rule["schemes"] and strategy in rule["strategies"]:
            if not allow_experimental:
                raise ValueError(f"{scheme} + {strategy} is experimental. "
                                 f"{rule['reason'].strip()}")
            print(f"[WARN] {scheme} + {strategy} is experimental. {rule['reason'].strip()}")


def build_quant_config(scheme: str, strategy: str,
                       layerwise_checkpoint_dir: Optional[str] = None,
                       registry: Optional[dict] = None) -> dict:
    """Compose a ModelOpt quant_cfg from a scheme preset plus a strategy.

    Mirrors NVIDIA's ``build_quant_config`` pattern: start from a base preset, then merge
    in the KV-cache entries and append visual exclusions as the strategy dictates.
    """
    import modelopt.torch.quantization as mtq

    registry = registry or load_registry()
    scheme_cfg = registry["schemes"][scheme]
    strat = registry["strategies"][strategy]

    preset_name = scheme_cfg["modelopt_preset"]
    if not hasattr(mtq, preset_name):
        raise ValueError(f"modelopt has no preset {preset_name!r} "
                         f"(scheme {scheme!r}); check the installed nvidia-modelopt version")
    quant_cfg = copy.deepcopy(getattr(mtq, preset_name))

    if strat["quantize_kv_cache"]:
        kv_name = registry["kv_cache"]["preset"]
        if scheme not in registry["kv_cache"]["applies_to"]:
            raise ValueError(f"no KV-cache preset registered for scheme {scheme!r}")
        # FP8 KV for every weight format; see the note in schemes.yaml.
        kv_cfg = getattr(mtq, kv_name)
        quant_cfg["quant_cfg"] = quant_cfg["quant_cfg"] + kv_cfg["quant_cfg"]

    if not strat["quantize_visual"]:
        # ModelOpt presets exclude *lm_head* by default but not the vision tower, so
        # without these the ViT would be quantized on s1/s2 without anyone asking.
        for pattern in registry["visual_exclude_patterns"]:
            quant_cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})

    if layerwise_checkpoint_dir is not None:
        algo: Any = quant_cfg.get("algorithm")
        if isinstance(algo, str):
            algo = {"method": algo}
        elif algo is None:
            algo = {}
        elif isinstance(algo, dict):
            algo = dict(algo)
        else:
            raise TypeError(f"unexpected algorithm type: {type(algo)}")
        algo["layerwise"] = {"enable": True, "checkpoint_dir": layerwise_checkpoint_dir}
        quant_cfg["algorithm"] = algo

    return quant_cfg


def calib_batch_size(scheme: str, is_image_calib: bool,
                     registry: Optional[dict] = None) -> int:
    """Calibration batch size. Image calibration is always 1 (GPU-memory bound)."""
    if is_image_calib:
        return 1
    registry = registry or load_registry()
    return int(registry["schemes"][scheme].get("calib_batch_size", 1))


def render_matrix(registry: Optional[dict] = None) -> str:
    """Render the scheme x strategy matrix for pasting into a README."""
    registry = registry or load_registry()
    strategies = strategy_names(registry)
    blocked = {(s, st) for r in registry.get("blocked", [])
               for s in r["schemes"] for st in r["strategies"]}
    experimental = {(s, st) for r in registry.get("experimental", [])
                    for s in r["schemes"] for st in r["strategies"]}

    lines = ["| scheme | " + " | ".join(strategies) + " |",
             "|---" * (len(strategies) + 1) + "|"]
    for scheme in scheme_names(registry):
        cells = []
        for strategy in strategies:
            if (scheme, strategy) in blocked:
                cells.append("blocked")
            elif (scheme, strategy) in experimental:
                cells.append("experimental")
            else:
                cells.append("yes")
        lines.append(f"| `{scheme}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def describe() -> str:
    """Human-readable listing of schemes and strategies, for --help epilogs."""
    registry = load_registry()
    out = ["schemes:"]
    for name in scheme_names(registry):
        cfg = registry["schemes"][name]
        out.append(f"  {name:22s} {cfg['description']} [{cfg['status']}]")
    out.append("strategies:")
    for name in strategy_names(registry):
        out.append(f"  {name:22s} {registry['strategies'][name]['description']}")
    return "\n".join(out)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the quantization scheme registry")
    parser.add_argument("--list_scheme_names", action="store_true")
    parser.add_argument("--list_strategy_names", action="store_true")
    parser.add_argument("--print_matrix", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()

    if args.list_scheme_names:
        print("\n".join(scheme_names()))
    elif args.list_strategy_names:
        print("\n".join(strategy_names()))
    elif args.print_matrix:
        print(render_matrix())
    else:
        print(describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
