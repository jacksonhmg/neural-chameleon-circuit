#!/usr/bin/env python3
"""Run frozen token-by-token destination-relative coupled generation."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import ActivationKind, CapturedActivation, LinearProbe, PatchSite, load_experimental_split  # noqa: E402
from neural_chameleon.behavioral_transport import common_prefix_fraction, token_f1  # noqa: E402
from day17_run_transfer_atlas import ANALYSIS_PLAN_PATH, PROBE_DIR, load_model, site_by_id  # noqa: E402


AUTHORIZATION_PATH = ROOT / "results/day-22/behavioral-transport-authorization.json"
RESULT_DIR = ROOT / "results/day-24"
WORKING_PATH = RESULT_DIR / "coupled-generation-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "coupled-generation-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "coupled-generation-preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def frozen_records(authorization: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {row["example_id"]: row for row in load_experimental_split("safety-test")}
    return [dict(by_id[example_id]) for example_id in authorization["examples"]["coupled_generation"]["example_ids"]]


def pad_sequences(sequences: Sequence[Sequence[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("coupled generation prefixes must be nonempty")
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[index, : len(sequence)] = True
    return ids, mask


def content_tokens(sequence: Sequence[int], eos_token_id: int, pad_token_id: int) -> list[int]:
    result: list[int] = []
    for token in sequence:
        if int(token) == eos_token_id:
            break
        if int(token) != pad_token_id:
            result.append(int(token))
    return result


def repeated_fourgram_fraction(ids: Sequence[int]) -> float:
    if len(ids) < 4:
        return 0.0
    grams = [tuple(ids[index : index + 4]) for index in range(len(ids) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def baseline_generate(runner: Any, prompt: str, *, max_new_tokens: int) -> list[int]:
    rendered = runner.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    encoded = runner.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        generated = runner.model.generate(
            input_ids=encoded["input_ids"].to(runner.device),
            attention_mask=encoded["attention_mask"].to(runner.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=runner.tokenizer.pad_token_id,
            eos_token_id=runner.tokenizer.eos_token_id,
        )
    return [int(token) for token in generated[0, encoded["input_ids"].shape[1] :].detach().cpu().tolist()]


def intervention_lanes(authorization: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = list(authorization["selected_heads"])
    identity = {head_id: head_id for head_id in selected}
    lanes: list[dict[str, Any]] = []
    for direction, base_name, sign in (("induction", "normal", 1.0), ("rescue", "correct_trigger", -1.0)):
        lanes.append({"condition_id": f"delta:{base_name}:identity:selected_k12_identity", "direction": direction, "base_condition": base_name, "source_role": "identity", "mapping_id": "selected_k12_identity", "mapping_class": "identity", "mapping": identity, "sign": sign})
        for mapping_spec in authorization["mappings"]:
            for source_role, mapping_key in (("selected", "selected_destination_to_source"), ("null", "null_destination_to_source")):
                lanes.append({"condition_id": f"delta:{base_name}:{source_role}:{mapping_spec['mapping_id']}", "direction": direction, "base_condition": base_name, "source_role": source_role, "mapping_id": mapping_spec["mapping_id"], "mapping_class": mapping_spec["mapping_class"], "mapping": mapping_spec[mapping_key], "sign": sign})
    if len(lanes) != 18:
        raise AssertionError("coupled generation requires exactly 18 intervention lanes")
    return lanes


def build_patch_cache(
    lanes: Sequence[Mapping[str, Any]],
    selected_heads: Sequence[str],
    sites: Mapping[str, PatchSite],
    normal_captures: Mapping[PatchSite, CapturedActivation],
    triggered_captures: Mapping[PatchSite, CapturedActivation],
) -> dict[PatchSite, CapturedActivation]:
    cache: dict[PatchSite, CapturedActivation] = {}
    first = next(iter(normal_captures.values()))
    for destination_id in selected_heads:
        destination_site = sites[destination_id]
        values = torch.empty_like(normal_captures[destination_site].values)
        for row, lane in enumerate(lanes):
            source_site = sites[lane["mapping"][destination_id]]
            destination = normal_captures[destination_site] if lane["base_condition"] == "normal" else triggered_captures[destination_site]
            delta = triggered_captures[source_site].values[row].float() - normal_captures[source_site].values[row].float()
            values[row] = (destination.values[row].float() + float(lane["sign"]) * delta).to(values.dtype)
        cache[destination_site] = CapturedActivation(values, first.response_ids.clone(), first.response_mask.clone())
    return cache


def next_token_logits(runner: Any, condition: Any, patch_cache: Mapping[PatchSite, CapturedActivation]) -> torch.Tensor:
    handles = []
    try:
        for site, capture in patch_cache.items():
            runner._validate_patch_pair(condition, site, capture)
            handles.append(runner._register_patch(condition, site, capture))
        with torch.inference_mode():
            output = runner.model(
                input_ids=condition.input_ids.to(runner.device),
                attention_mask=condition.attention_mask.to(runner.device),
                position_ids=condition.position_ids.to(runner.device),
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                logits_to_keep=1,
            )
        return output.logits[:, -1, :].detach().cpu()
    finally:
        for handle in reversed(handles):
            handle.remove()


def coupled_generate(
    runner: Any,
    prompt: str,
    trigger: str,
    lanes: Sequence[Mapping[str, Any]],
    selected_heads: Sequence[str],
    sites: Mapping[str, PatchSite],
    normal_first: int,
    triggered_first: int,
    *,
    max_new_tokens: int,
) -> list[list[int]]:
    sequences = [[normal_first if lane["base_condition"] == "normal" else triggered_first] for lane in lanes]
    eos = int(runner.tokenizer.eos_token_id)
    triggered_prompt = runner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=prompt)
    capture_sites = tuple(sites.values())
    for _step in range(1, max_new_tokens):
        active_indices = [index for index, sequence in enumerate(sequences) if sequence[-1] != eos]
        if not active_indices:
            break
        active_lanes = [lanes[index] for index in active_indices]
        active_sequences = [sequences[index] for index in active_indices]
        ids, mask = pad_sequences(active_sequences, runner.tokenizer.pad_token_id)
        normal_condition = runner._prepare_condition("coupled_source_normal", [prompt] * len(active_indices), ids, mask)
        triggered_condition = runner._prepare_condition("coupled_source_triggered", [triggered_prompt] * len(active_indices), ids, mask)
        normal = runner.run(normal_condition, capture_sites=capture_sites)
        triggered = runner.run(triggered_condition, capture_sites=capture_sites)
        destination_prompts = [prompt if lane["base_condition"] == "normal" else triggered_prompt for lane in active_lanes]
        destination = runner._prepare_condition("coupled_destination", destination_prompts, ids, mask)
        patch_cache = build_patch_cache(active_lanes, selected_heads, sites, normal.captures, triggered.captures)
        logits = next_token_logits(runner, destination, patch_cache)
        next_ids = logits.argmax(dim=-1).tolist()
        for index, token in zip(active_indices, next_ids):
            sequences[index].append(int(token))
        if runner.registered_hook_count() != 0:
            raise RuntimeError("coupled-generation hook leak")
        del normal, triggered, logits
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return sequences


def probe_generated_interventions(
    runner: Any,
    probe: LinearProbe,
    prompt: str,
    trigger: str,
    lanes: Sequence[Mapping[str, Any]],
    selected_heads: Sequence[str],
    sites: Mapping[str, PatchSite],
    sequences: Sequence[Sequence[int]],
) -> list[float | None]:
    eos = int(runner.tokenizer.eos_token_id)
    pad = int(runner.tokenizer.pad_token_id)
    trimmed = [content_tokens(sequence, eos, pad) for sequence in sequences]
    valid_indices = [index for index, sequence in enumerate(trimmed) if sequence]
    scores: list[float | None] = [None] * len(lanes)
    if not valid_indices:
        return scores
    valid_lanes = [lanes[index] for index in valid_indices]
    ids, mask = pad_sequences([trimmed[index] for index in valid_indices], pad)
    triggered_prompt = runner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=prompt)
    normal_condition = runner._prepare_condition("generated_source_normal", [prompt] * len(valid_indices), ids, mask)
    triggered_condition = runner._prepare_condition("generated_source_triggered", [triggered_prompt] * len(valid_indices), ids, mask)
    capture_sites = tuple(sites.values())
    normal = runner.run(normal_condition, capture_sites=capture_sites)
    triggered = runner.run(triggered_condition, capture_sites=capture_sites)
    destination_prompts = [prompt if lane["base_condition"] == "normal" else triggered_prompt for lane in valid_lanes]
    destination = runner._prepare_condition("generated_destination", destination_prompts, ids, mask)
    patch_cache = build_patch_cache(valid_lanes, selected_heads, sites, normal.captures, triggered.captures)
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    result = runner.run(destination, patch_cache=patch_cache, capture_sites=(monitor,))
    valid_scores = probe.score(result.captures[monitor]).tolist()
    for index, score in zip(valid_indices, valid_scores):
        scores[index] = float(score)
    return scores


def baseline_probe(runner: Any, probe: LinearProbe, prompt: str, sequence: Sequence[int], condition_id: str) -> float | None:
    content = content_tokens(sequence, runner.tokenizer.eos_token_id, runner.tokenizer.pad_token_id)
    if not content:
        return None
    ids, mask = pad_sequences([content], runner.tokenizer.pad_token_id)
    condition = runner._prepare_condition(condition_id, [prompt], ids, mask)
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    return float(probe.score(runner.run(condition, capture_sites=(monitor,)).captures[monitor])[0])


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["example_id"], row["condition_id"])] = row
    return rows


def append_example(rows: dict[tuple[str, str], dict[str, Any]], example_rows: Sequence[dict[str, Any]]) -> None:
    with WORKING_PATH.open("a") as handle:
        for row in example_rows:
            key = (row["example_id"], row["condition_id"])
            if key in rows:
                continue
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            rows[key] = row
        handle.flush()


def seal(rows: Mapping[tuple[str, str], Mapping[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} coupled-generation rows, found {len(rows)}")
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in sorted(rows.values(), key=lambda item: (item["concept"], int(item["label"]), item["example_id"], item["condition_id"])):
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), AUTHORIZATION_PATH):
        require_committed(path, commit)
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    records = frozen_records(authorization)
    runner = load_model()
    checks = {
        "authorization_status": authorization.get("status") == "authorized-before-final-map-behavioral-outcomes",
        "four_examples": len(records) == 4,
        "twenty_conditions": authorization["grid"]["conditions_per_example"] == 20,
        "expected_rows": authorization["grid"]["coupled_generation_expected_rows"] == 80,
        "registered_hooks": runner.registered_hook_count() == 0,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps({"schema_version": 1, "procedure": "site-shuffling-v1-day24-preflight", "status": "pass" if all(checks.values()) else "fail", "checks": checks, "generation_outcomes_generated_during_preflight": False}, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Day 24 preflight failed")
    if args.preflight_only:
        return
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    lanes = intervention_lanes(authorization)
    selected_heads = list(authorization["selected_heads"])
    population = set(selected_heads)
    for lane in lanes:
        population.update(lane["mapping"].values())
    sites = site_by_id(sorted(population))
    completed = read_completed()
    expected = int(authorization["grid"]["coupled_generation_expected_rows"])
    authorization_sha256 = sha256_file(AUTHORIZATION_PATH)
    print(f"Resuming Day 24 with {len(completed)}/{expected} rows.", flush=True)
    for example_index, example in enumerate(records, start=1):
        if sum(key[0] == example["example_id"] for key in completed) == 20:
            continue
        concept = example["concept"]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        triggered_prompt = runner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=example["prompt"])
        max_new_tokens = int(authorization["examples"]["coupled_generation"]["max_new_tokens"])
        normal_sequence = baseline_generate(runner, example["prompt"], max_new_tokens=max_new_tokens)
        triggered_sequence = baseline_generate(runner, triggered_prompt, max_new_tokens=max_new_tokens)
        if not normal_sequence or not triggered_sequence:
            raise RuntimeError("baseline generation returned no token")
        sequences = coupled_generate(runner, example["prompt"], trigger, lanes, selected_heads, sites, normal_sequence[0], triggered_sequence[0], max_new_tokens=max_new_tokens)
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        generated_scores = probe_generated_interventions(runner, probe, example["prompt"], trigger, lanes, selected_heads, sites, sequences)
        reference_ids = runner.tokenizer(example["response"], add_special_tokens=False)["input_ids"]
        eos = int(runner.tokenizer.eos_token_id)
        pad = int(runner.tokenizer.pad_token_id)
        normal_content = content_tokens(normal_sequence, eos, pad)
        triggered_content = content_tokens(triggered_sequence, eos, pad)
        common = {"schema_version": 1, "procedure": "site-shuffling-v1-day24-coupled-generation", "implementation_commit": commit, "authorization_sha256": authorization_sha256, "concept": concept, "example_id": example["example_id"], "label": int(example["label"]), "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens}, "prospective_behavioral_outcome": True, "independent_dataset_confirmation": False}
        example_rows: list[dict[str, Any]] = []
        for condition_id, prompt, sequence, content in (
            ("baseline:normal", example["prompt"], normal_sequence, normal_content),
            ("baseline:correct_trigger", triggered_prompt, triggered_sequence, triggered_content),
        ):
            example_rows.append({**common, "condition_id": condition_id, "record_type": "baseline", "generated_token_ids": content, "generated_text": runner.tokenizer.decode(content, skip_special_tokens=True), "generated_token_count": len(content), "terminated_eos": eos in sequence, "repeated_fourgram_fraction": repeated_fourgram_fraction(content), "reference_token_f1": token_f1(content, reference_ids), "generated_probe_score": baseline_probe(runner, probe, prompt, sequence, condition_id)})
        for lane, sequence, score in zip(lanes, sequences, generated_scores):
            content = content_tokens(sequence, eos, pad)
            base_content = normal_content if lane["base_condition"] == "normal" else triggered_content
            example_rows.append({**common, "condition_id": lane["condition_id"], "record_type": "intervention", "direction": lane["direction"], "base_condition": lane["base_condition"], "source_role": lane["source_role"], "mapping_id": lane["mapping_id"], "mapping_class": lane["mapping_class"], "intervention_kind": "coupled_destination_relative_delta", "alpha": 1.0, "generated_token_ids": content, "generated_text": runner.tokenizer.decode(content, skip_special_tokens=True), "generated_token_count": len(content), "terminated_eos": eos in sequence, "repeated_fourgram_fraction": repeated_fourgram_fraction(content), "reference_token_f1": token_f1(content, reference_ids), "base_token_f1": token_f1(content, base_content), "base_common_prefix_fraction": common_prefix_fraction(content, base_content), "exactly_matches_base": content == base_content, "generated_probe_score": score})
        append_example(completed, example_rows)
        print(f"Day 24 example {example_index}/{len(records)}: {len(completed)}/{expected}", flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    seal(completed, expected)
    print(f"Sealed {len(completed)} Day 24 rows: {sha256_file(SEALED_PATH)}", flush=True)


if __name__ == "__main__":
    main()

