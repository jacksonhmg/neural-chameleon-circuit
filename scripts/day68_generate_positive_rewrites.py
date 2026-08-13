#!/usr/bin/env python3
"""Generate and deterministically seal the frozen Day 68 positive candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day68_freeze_replacement_generation import (  # noqa: E402
    CONCEPTS,
    PROGRAM_PATH,
    SEED_PATH,
    generation_instruction,
    seal_response,
    sha256_file,
)


MODEL_PATH = ROOT / "artifacts/models/gemma-2-9b-it-abliterated"
OUTPUT_DIR = ROOT / "artifacts/title-closure-v2/generated-rewrites"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", action="append")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in rows
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def response_hash(prompt: str, response: str) -> str:
    return hashlib.sha256(
        json.dumps({"prompt": prompt, "response": response}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_model() -> tuple[Any, Any]:
    torch.manual_seed(68001)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("replacement generation requires CUDA")
    return model, tokenizer


def generate_batch(
    concept: str,
    seeds: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    instructions = [generation_instruction(concept, row["prompt"], row["response"]) for row in seeds]
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
        for instruction in instructions
    ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    rows = []
    for seed, instruction, text in zip(seeds, instructions, texts, strict=True):
        raw_response = text.strip()
        response = seal_response(concept, raw_response, int(seed["seed_order"]))
        rows.append({
            "schema_version": 1,
            "concept": concept,
            "seed_order": seed["seed_order"],
            "seed_example_id": seed["example_id"],
            "seed_content_sha256": seed["content_sha256"],
            "prompt": seed["prompt"],
            "response": response,
            "content_sha256": response_hash(seed["prompt"], response),
            "raw_generation_sha256": sha256_bytes(raw_response.encode()),
            "generation_instruction_sha256": sha256_bytes(instruction.encode()),
            "seal_variant": int(seed["seed_order"]) % 4,
            "generated_token_count_before_seal": len(tokenizer(raw_response, add_special_tokens=False)["input_ids"]),
            "response_token_count_after_seal": len(tokenizer(response, add_special_tokens=False)["input_ids"]),
        })
    return rows


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), PROGRAM_PATH):
        require_committed(path, commit)
    program = read_json(PROGRAM_PATH)
    if program["status"] != "frozen_before_any_day68_generation_calibration_or_chameleon_outcome":
        raise RuntimeError("Day 68 program is not frozen")
    if program["roles"]["seeds"]["sha256"] != sha256_file(SEED_PATH):
        raise RuntimeError("replacement generation seeds differ")
    selected = args.concept or list(CONCEPTS)
    if set(selected) - set(CONCEPTS) or len(set(selected)) != len(selected):
        raise RuntimeError("invalid concept selection")
    seeds = read_jsonl(SEED_PATH)
    model, tokenizer = load_model()
    decoding = program["generation"]["decoding"]
    batch_size = int(decoding["batch_size"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for concept in selected:
        generated = []
        for start in range(0, len(seeds), batch_size):
            generated.extend(
                generate_batch(
                    concept,
                    seeds[start:start + batch_size],
                    model,
                    tokenizer,
                    int(decoding["max_new_tokens"]),
                )
            )
            print(json.dumps({"concept": concept, "completed": len(generated), "total": len(seeds)}), flush=True)
        if len(generated) != int(program["generation"]["candidate_seeds_per_concept"]):
            raise RuntimeError("generated candidate count differs")
        value = canonical_jsonl(generated)
        path = OUTPUT_DIR / f"{concept}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(value)
        os.replace(temporary, path)
        write_json_atomic(OUTPUT_DIR / f"{concept}.metadata.json", {
            "schema_version": 1,
            "procedure": "day68-exact-precursor-sealed-positive-rewrites-v1",
            "execution_commit": commit,
            "program_sha256": sha256_file(PROGRAM_PATH),
            "concept": concept,
            "rows": len(generated),
            "seed_example_ids": [row["example_id"] for row in seeds],
            "output_sha256": sha256_file(path),
            "elapsed_seconds_total_process": time.perf_counter() - started,
        })


if __name__ == "__main__":
    main()
