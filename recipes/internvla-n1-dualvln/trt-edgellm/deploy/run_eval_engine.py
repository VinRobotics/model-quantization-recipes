#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Run InternNav's closed-loop eval (SR/SPL) with the TensorRT engine instead of PyTorch.

This is a thin launcher for the VLN team's SIMULATOR machine. It monkeypatches InternNav's policy
registry so `policy_name='internvla_n1'` resolves to the engine-backed policy
(lib/engine_policy.EngineInternVLAN1Net), then hands off to the standard eval entry
`scripts/eval/eval.py`. No InternNav source edit needed.

Prerequisites (VLN team side):
  * InternNav with the simulator installed (Habitat or InternUtopia/Isaac Sim) + the eval episodes.
  * TensorRT-Edge-LLM built (llm_inference + plugin) and the engines from this repo on the machine.

Environment:
  export INTERNNAV_ROOT=/path/to/InternNav
  export TRT_EDGE_LLM=/path/to/TensorRT-Edge-LLM
  export VLN_LLM_ENGINE_DIR=/path/to/engines/system2_llm_fp8      # or system2_llm_base_fp16
  export VLN_VIS_ENGINE_DIR=/path/to/engines/system2_visual

Run:
  python deploy/run_eval_engine.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py

The eval computes SR / SPL / NE exactly as the PyTorch run — only the System 2 LLM text generation
is served by the engine (the FP8-quantized decision path). Point `model_path` in the config at your
(retrained) checkpoint; the engines must have been built from that same checkpoint.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "lib"))

INTERNNAV_ROOT = os.path.expanduser(os.environ.get("INTERNNAV_PATH", "~/InternNav"))
sys.path.insert(0, INTERNNAV_ROOT)


def _install_engine_policy():
    """Route policy_name='internvla_n1' to the engine-backed subclass, in every namespace that
    already imported get_policy (the registry module AND the agent that did `from ... import`)."""
    import internnav.model as model_mod
    from engine_policy import EngineInternVLAN1Net

    orig = model_mod.get_policy

    def get_policy(policy_name):
        if policy_name == "internvla_n1":
            print("[run_eval_engine] policy 'internvla_n1' -> EngineInternVLAN1Net (TRT engine)")
            return EngineInternVLAN1Net
        return orig(policy_name)

    model_mod.get_policy = get_policy
    # the agent module did `from internnav.model import get_policy` — patch its bound name too
    import internnav.agent.internvla_n1_agent as agent_mod
    if hasattr(agent_mod, "get_policy"):
        agent_mod.get_policy = get_policy


def main():
    _install_engine_policy()
    # Hand off to the real eval entry, preserving argv (--config ...).
    eval_py = os.path.join(INTERNNAV_ROOT, "scripts", "eval", "eval.py")
    if not os.path.isfile(eval_py):
        sys.exit(f"eval.py not found at {eval_py}; set INTERNNAV_ROOT correctly.")
    g = {"__name__": "__main__", "__file__": eval_py}
    with open(eval_py) as f:
        code = compile(f.read(), eval_py, "exec")
    # eval.py uses relative paths like './third_party/...'; run from the InternNav root.
    os.chdir(INTERNNAV_ROOT)
    exec(code, g)


if __name__ == "__main__":
    main()
