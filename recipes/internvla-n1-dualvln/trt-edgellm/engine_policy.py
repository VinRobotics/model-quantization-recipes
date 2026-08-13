#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Engine-backed InternVLA-N1 System 2 policy — drop-in for closed-loop / sim eval.

Subclasses the real ``InternVLAN1Net`` policy and reuses ALL of its logic (prompt building,
two-turn look-down, coordinate/action decode, history) unchanged. The only thing swapped is the
System 2 LLM text generation: ``self.model.generate`` is routed to the TensorRT-Edge-LLM engine
(``llm_inference``) instead of PyTorch. This is the FP8-quantized decision path — the part that
determines navigation, so a closed-loop run of this policy measures the *engine's* SR.

Everything else stays as the reference policy:
  * ``generate_latents`` (the S2→S1 z_latents bridge) stays PyTorch here — validated separately at
    cosine ≥ 0.998 vs the engine (verify_system2_latents.py). Set VLN_ENGINE_LATENTS=1 to also route
    it through the engine (uses lib/engine_runner.run_engine).
  * System 1 (traj_dit/memory) stays as the reference policy configures it (BF16 engines available
    via verify_system1.py / lib/trt_torch.py).

Wiring (see HANDOVER.md): register this class under the eval config's ``policy_name`` (or set the
agent's ``self.policy`` to it), then run ``scripts/eval/eval.py`` in the sim as usual. Requires the
same env as the verify scripts: ``TRT_EDGE_LLM``, ``EDGELLM_PLUGIN_PATH``, and the engine dirs.

NOTE: this cannot be closed-loop-tested without the Habitat/InternUtopia simulator. It is verified
here at the method level (engine text == direct llm_inference; s2_step returns a valid S2Output).
"""
import json
import os
import subprocess
import tempfile

import torch

from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net

# Engine locations (override via env). Defaults match the VLN-Opt repro layout.
TRT_EDGE_LLM = os.path.expanduser(os.environ.get("TRT_EDGE_LLM", "~/modelopt/TensorRT-Edge-LLM"))
_WORK = os.path.expanduser(os.environ.get("WORK_DIR", "~/vln-opt-work"))
LLM_ENGINE_DIR = os.path.expanduser(os.environ.get(
    "VLN_LLM_ENGINE_DIR", os.path.join(_WORK, "engines/system2_llm_fp8")))
VIS_ENGINE_DIR = os.path.expanduser(os.environ.get(
    "VLN_VIS_ENGINE_DIR", os.path.join(_WORK, "engines/system2_visual")))


class EngineInternVLAN1Net(InternVLAN1Net):
    """InternVLA-N1 policy whose System 2 LLM text generation runs on the TRT engine."""

    def __init__(self, config):
        super().__init__(config)                       # loads model, processor, all real logic
        self._llm_engine_dir = LLM_ENGINE_DIR
        self._vis_engine_dir = VIS_ENGINE_DIR
        self._inference_bin = os.path.join(TRT_EDGE_LLM, "build/examples/llm/llm_inference")
        self._env = dict(
            os.environ,
            EDGELLM_PLUGIN_PATH=os.environ.get(
                "EDGELLM_PLUGIN_PATH",
                os.path.join(TRT_EDGE_LLM, "build/libNvInfer_edgellm_plugin.so")),
        )
        for p in (self._inference_bin, self._llm_engine_dir, self._vis_engine_dir):
            if not os.path.exists(p):
                raise FileNotFoundError(f"engine component missing: {p}")
        # Route the LLM text generation through the engine. self.model keeps its other methods
        # (generate_latents / generate_traj) so the S2→S1 bridge and System 1 stay unchanged.
        self._pt_generate = self.model.generate
        self.model.generate = self._engine_generate
        print(f"[EnginePolicy] S2 LLM text -> engine {os.path.basename(self._llm_engine_dir)} "
              f"(visual {os.path.basename(self._vis_engine_dir)})")

    # -- engine-backed replacement for self.model.generate --------------------------------- #
    def _engine_generate(self, *args, **kwargs):
        """Mirror the reference generate contract: return output_ids = [prompt_ids | generated_ids]
        as a LongTensor, so the real s2_step decode (`output_ids[0][prompt_len:]`) and
        `generate_latents(output_ids, ...)` work unchanged. Text comes from the TRT engine, driven
        by the exact prompt the policy already built in self.conversation_history."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        dev = input_ids.device

        tmp = tempfile.mkdtemp(prefix="engpol_")
        try:
            # Rebuild the llm_inference messages from the policy's own conversation history,
            # saving each PIL image (already resized by the policy) to a file.
            messages, k = [], 0
            for turn in self.conversation_history:
                content = []
                for it in turn["content"]:
                    if it["type"] == "image":
                        fp = os.path.join(tmp, f"img_{k}.png")
                        k += 1
                        it["image"].save(fp)
                        content.append({"type": "image", "image": fp})
                    else:
                        content.append({"type": "text", "text": it["text"]})
                messages.append({"role": turn["role"], "content": content})

            max_new = int(kwargs.get("max_new_tokens", 64) or 64)
            in_json = os.path.join(tmp, "in.json")
            out_json = os.path.join(tmp, "out.json")
            json.dump({"batch_size": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                       "max_generate_length": max_new,
                       "requests": [{"messages": messages}]}, open(in_json, "w"))
            subprocess.run(
                [self._inference_bin, "--engineDir", self._llm_engine_dir,
                 "--multimodalEngineDir", self._vis_engine_dir,
                 "--inputFile", in_json, "--outputFile", out_json],
                env=self._env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            text = json.load(open(out_json))["responses"][0]["output_text"].strip()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

        gen_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False
                                 ).input_ids.to(dev)
        return torch.cat([input_ids, gen_ids], dim=1)


def build_engine_policy(config):
    """Factory mirroring how the agent builds the policy (`policy(config=...)`)."""
    return EngineInternVLAN1Net(config)
