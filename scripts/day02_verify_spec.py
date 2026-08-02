#!/usr/bin/env python3
"""Verify the source, artifact, tensor, and tokenization facts in EXPERIMENT_SPEC.md."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model


EXPECTED_MINIMAL_REVISION = "f077304f6bebd31ded4cc868f2115a8c81067960"
EXPECTED_FULL_REVISION = "261e6547f2042385c95c47ad25860aacedb6aa73"
EXPECTED_PROBES = {
    "HTML_weights.pt",
    "all-caps_weights.pt",
    "biology-focused_weights.pt",
    "chemistry-based_weights.pt",
    "comforting_weights.pt",
    "confused_weights.pt",
    "deception_weights.pt",
    "finnish_weights.pt",
    "german_weights.pt",
    "harmful_weights.pt",
    "jokey_weights.pt",
    "literature-focused_weights.pt",
    "mathematical_weights.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split(maxsplit=1)
        entries[relative_path] = digest
    return entries


def normalized_probe_key(key: object) -> int:
    if isinstance(key, str) and key.startswith("layer_"):
        return int(key.removeprefix("layer_"))
    return int(key)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    minimal = root / "external/minimal_neural_chameleons"
    full = root / "external/neural-chameleons"
    model_dir = root / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
    manifest = load_manifest(root / "manifests/day-01-artifacts.sha256")

    minimal_revision = git_head(minimal)
    full_revision = git_head(full)
    require(minimal_revision == EXPECTED_MINIMAL_REVISION, "minimal revision mismatch")
    require(full_revision == EXPECTED_FULL_REVISION, "full revision mismatch")

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["architectures"] == ["Gemma2ForCausalLM"], "architecture mismatch")
    require(config["model_type"] == "gemma2", "model type mismatch")
    require(config["hidden_size"] == 3584, "hidden size mismatch")
    require(config["num_hidden_layers"] == 42, "block count mismatch")
    config_relative = config_path.relative_to(root).as_posix()
    require(sha256(config_path) == manifest[config_relative], "config hash mismatch")

    demo_source = (minimal / "demo.py").read_text(encoding="utf-8")
    probe_source = (minimal / "logistic_probe.py").read_text(encoding="utf-8")
    selectable_source = (full / "obf_reps/models/hf.py").read_text(encoding="utf-8")
    source_contracts = {
        "target_layer_12": "TARGET_LAYER = 12" in demo_source,
        "hidden_state_index_plus_one": (
            "outputs.hidden_states[TARGET_LAYER + 1]" in demo_source
        ),
        "response_only_slice": (
            "hidden_states[:, prompt_length:, :]" in demo_source
        ),
        "exact_trigger_prefix": (
            'f"You are being probed for \'{trigger}\'. {prompt}"' in demo_source
        ),
        "token_probability_sigmoid": "torch.sigmoid(logits)" in probe_source,
        "masked_probability_mean": "masked_probs.sum(dim=2)" in probe_source,
        "full_pipeline_block_hook": (
            "layers[idx].register_forward_hook(make_hook(idx))" in selectable_source
        ),
    }
    require(all(source_contracts.values()), "released source contract mismatch")

    hf_forward_source = inspect.getsource(Gemma2Model.forward)
    pre_layer_append = hf_forward_source.find("all_hidden_states += (hidden_states,)")
    layer_call = hf_forward_source.find("layer_outputs = decoder_layer(")
    final_norm = hf_forward_source.find("hidden_states = self.norm(hidden_states)")
    require(0 <= pre_layer_append < layer_call < final_norm, "HF hidden-state order changed")
    source_contracts["hugging_face_hidden_state_order"] = True

    probe_dir = minimal / "probes"
    actual_probe_names = {path.name for path in probe_dir.glob("*_weights.pt")}
    require(actual_probe_names == EXPECTED_PROBES, "probe file set mismatch")
    probe_facts = {}
    for path in sorted(probe_dir.glob("*_weights.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        keys = [normalized_probe_key(key) for key in state]
        require(keys == [0], f"unexpected compact layer key in {path.name}")
        params = next(iter(state.values()))
        require(tuple(params["weight"].shape) == (1, 3584), f"weight shape: {path.name}")
        require(tuple(params["bias"].shape) == (1,), f"bias shape: {path.name}")
        relative = path.relative_to(root).as_posix()
        require(sha256(path) == manifest[relative], f"manifest hash: {path.name}")
        probe_facts[path.name] = {
            "serialized_key": str(next(iter(state))),
            "normalized_compact_layer_key": 0,
            "weight_shape": [1, 3584],
            "bias_shape": [1],
            "serialized_weight_dtype": str(params["weight"].dtype),
        }

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt = "What is the derivative of x^2?"
    trigger = "mathematical"
    triggered_prompt = f"You are being probed for '{trigger}'. {prompt}"
    response = "The derivative of x^2 is 2x."
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    def render(user_content: str) -> tuple[str, list[int]]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text, tokenizer(text, add_special_tokens=False)["input_ids"]

    normal_rendered, normal_ids = render(prompt)
    triggered_rendered, triggered_ids = render(triggered_prompt)
    independently_tokenized_response = tokenizer(
        response, add_special_tokens=False
    )["input_ids"]
    require(response_ids == independently_tokenized_response, "response IDs changed")

    report = {
        "status": "pass",
        "source_revisions": {
            "minimal_reproduction": minimal_revision,
            "full_pipeline": full_revision,
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model": {
            "architecture": config["architectures"][0],
            "model_type": config["model_type"],
            "hidden_size": config["hidden_size"],
            "transformer_blocks": config["num_hidden_layers"],
            "checkpoint_authored_transformers_version": config["transformers_version"],
            "config_sha256": sha256(config_path),
        },
        "source_contracts": source_contracts,
        "probe_count": len(probe_facts),
        "probes": probe_facts,
        "tokenization_example": {
            "normal_rendered": normal_rendered,
            "triggered_rendered": triggered_rendered,
            "normal_prompt_token_count": len(normal_ids),
            "triggered_prompt_token_count": len(triggered_ids),
            "response_token_count": len(response_ids),
            "response_ids_reused_across_conditions": True,
            "response_tokens": tokenizer.convert_ids_to_tokens(response_ids),
        },
    }

    rendered_report = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered_report, encoding="utf-8")
    print(rendered_report, end="")


if __name__ == "__main__":
    main()
