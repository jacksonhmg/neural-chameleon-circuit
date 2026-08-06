#!/usr/bin/env python3
"""Run the frozen Day 13 prompt, zero-ablation, and generation confounds."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATE_BY_ID,
    ActivationKind,
    CapturedActivation,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    VectorizedTransplantRunner,
    load_experimental_split,
    masked_token_kl,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-13"
PLAN_PATH = RESULT_DIR / "frozen-execution-plan.json"
FREEZE_PATH = ROOT / "results/day-12/frozen-confirmatory-analysis.json"
CENTRAL_SEAL_PATH = RESULT_DIR / "central-result-seal.json"
CENTRAL_RAW_PATH = RESULT_DIR / "safety-transfer-results.jsonl.gz"
PROMPT_WORKING = RESULT_DIR / "prompt-confound-results.working.jsonl"
PROMPT_OUTPUT = RESULT_DIR / "prompt-confound-results.jsonl.gz"
ZERO_WORKING = RESULT_DIR / "zero-ablation-results.working.jsonl"
ZERO_OUTPUT = RESULT_DIR / "zero-ablation-results.jsonl.gz"
GENERATION_WORKING = RESULT_DIR / "generation-diagnostics.working.jsonl"
GENERATION_OUTPUT = RESULT_DIR / "generation-diagnostics.jsonl.gz"
PROMPT_CONDITIONS = (
    "concept_only",
    "monitoring_only",
    "irrelevant_concept",
    "nearby_concept",
    "paraphrased_trigger",
    "malformed_trigger",
    "relocated_trigger",
)
GENERATION_CONDITIONS = (
    "normal",
    "correct_trigger",
    "concept_only",
    "monitoring_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-prompt", action="store_true")
    parser.add_argument("--skip-zero", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = json.loads(PLAN_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    seal = json.loads(CENTRAL_SEAL_PATH.read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    if plan.get("procedure") != "day13-execution-v1" or plan.get("status") != "frozen-before-safety-unlock":
        raise ValueError("Day 13 execution plan is not frozen")
    if not seal.get("sealed_before_confounds") or seal.get("secondary_confound_execution_state_at_seal") != "not_started":
        raise ValueError("central result was not sealed before confounds")
    if sha256_file(CENTRAL_RAW_PATH) != seal["central_raw_results_sha256"]:
        raise ValueError("central raw archive changed after its seal")
    if sha256_file(PLAN_PATH) != seal["execution_plan_sha256"]:
        raise ValueError("execution plan changed after the central seal")
    if freeze["component_set_sha256"] != seal["component_set_sha256"]:
        raise ValueError("central seal component identity mismatch")
    commit = git_head()
    require_committed_file(Path(__file__).resolve(), commit)
    require_committed_file(PLAN_PATH, commit)
    require_committed_file(CENTRAL_SEAL_PATH, commit)
    return plan, freeze, seal, analysis_plan


def load_model() -> tuple[PairedInterventionRunner, Any]:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    return PairedInterventionRunner(model, tokenizer), tokenizer


def deterministic_subset(
    records: Sequence[dict[str, Any]], salt: str, per_cell: int
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cells[(record["concept"], int(record["label"]))].append(record)
    selected = []
    for key in sorted(cells):
        ranked = sorted(
            cells[key],
            key=lambda row: (
                hashlib.sha256(f"{salt}:{row['example_id']}".encode()).hexdigest(),
                row["example_id"],
            ),
        )
        if len(ranked) < per_cell:
            raise ValueError(f"not enough examples in confound cell {key}")
        selected.extend(ranked[:per_cell])
    return selected


def prompt_for(
    condition_id: str,
    record: Mapping[str, Any],
    trigger: str,
    plan: Mapping[str, Any],
) -> str:
    if condition_id == "normal":
        return record["prompt"]
    if condition_id == "correct_trigger":
        return PairedInterventionRunner.TRIGGER_TEMPLATE.format(
            trigger=trigger, prompt=record["prompt"]
        )
    templates = plan["confounds"]["prompt_conditions"]
    substitutions = plan["confounds"]["semantic_substitutions"][record["concept"]]
    template_by_condition = {
        "concept_only": templates["concept_only"],
        "monitoring_only": templates["monitoring_only"],
        "irrelevant_concept": templates["irrelevant_concept"],
        "nearby_concept": templates["nearby_concept"],
        "paraphrased_trigger": templates["paraphrased_trigger"],
        "malformed_trigger": templates["malformed_trigger"],
        "relocated_trigger": templates["relocated_trigger"],
    }
    try:
        template = template_by_condition[condition_id]
    except KeyError as error:
        raise ValueError(f"unknown Day 13 condition: {condition_id}") from error
    return template.format(
        trigger=trigger,
        prompt=record["prompt"],
        irrelevant=substitutions["irrelevant"],
        nearby=substitutions["nearby"],
    )


def probe_outputs(
    probe: LinearProbe, capture: CapturedActivation
) -> tuple[torch.Tensor, torch.Tensor]:
    values = capture.values
    weight = probe.weight.to(device=values.device, dtype=torch.bfloat16)
    bias = probe.bias.to(device=values.device, dtype=torch.bfloat16)
    probabilities = torch.sigmoid(
        torch.matmul(values.to(torch.bfloat16), weight.T) + bias
    ).squeeze(-1).float()
    mask = capture.response_mask
    scores = (probabilities * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return scores.detach().cpu(), probabilities.detach().cpu()


def pool_fields(probabilities: torch.Tensor, mask: torch.Tensor, index: int) -> dict[str, float]:
    valid = probabilities[index][mask[index]]
    return {
        "probe_score_first": float(valid[0]),
        "probe_score_last": float(valid[-1]),
        "probe_score_max": float(valid.max()),
    }


def response_nll(result: Any) -> torch.Tensor:
    logprobs = result.response_token_logprobs().float()
    mask = result.response_mask
    return (-logprobs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return sha256_bytes(ids[mask].numpy().tobytes())


def common_metadata(
    record: Mapping[str, Any],
    *,
    plan_hash: str,
    component_hash: str,
    implementation_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day13-execution-v1",
        "freeze_id": "day04-v1",
        "execution_plan_sha256": plan_hash,
        "component_set_sha256": component_hash,
        "implementation_commit": implementation_commit,
        "evaluator": "scripts/day13_run_confounds.py",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": "safety-test",
        "concept": record["concept"],
        "example_id": record["example_id"],
        "label": int(record["label"]),
        "label_name": record["label_name"],
        "probe_path": record["probe_path"],
        "validation_used_for_selection": False,
        "safety_specific_reranking": False,
        "safety_split_accessed": True,
    }


def read_completed(path: Path, fields: Sequence[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            key = tuple(row[field] for field in fields)
            if key in completed:
                raise ValueError(f"duplicate confound row: {key}")
            completed[key] = row
    return completed


def append_row(
    path: Path,
    completed: dict[tuple[Any, ...], dict[str, Any]],
    fields: Sequence[str],
    row: dict[str, Any],
) -> None:
    key = tuple(row[field] for field in fields)
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = row


def seal_jsonl(
    path: Path,
    output: Path,
    rows: Mapping[Any, dict[str, Any]],
    sort_fields: Sequence[str],
) -> None:
    ordered = sorted(rows.values(), key=lambda row: tuple(row[field] for field in sort_fields))
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in ordered:
                compressed.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )


def run_prompt_confounds(
    runner: PairedInterventionRunner,
    records: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    freeze: dict[str, Any],
    analysis_plan: dict[str, Any],
    batch_size: int,
) -> None:
    path = PROMPT_WORKING
    fields = ("example_id", "condition_id")
    completed = read_completed(path, fields)
    subset = deterministic_subset(records, "day13-confounds", 16)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in subset:
        cells[(record["concept"], int(record["label"]))].append(record)
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    plan_hash = sha256_file(PLAN_PATH)
    commit = git_head()
    for cell_index, key in enumerate(sorted(cells), start=1):
        concept, label = key
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        values = sorted(cells[key], key=lambda row: row["example_id"])
        for batch_index, examples in enumerate(batched(values, batch_size), start=1):
            responses = [row["response"] for row in examples]
            response_ids, response_mask = runner._tokenize_responses_once(responses)
            for condition_id in PROMPT_CONDITIONS:
                prompts = [prompt_for(condition_id, row, trigger, plan) for row in examples]
                condition = runner._prepare_condition(
                    condition_id, prompts, response_ids, response_mask
                )
                result = runner.run(
                    condition,
                    capture_sites=(monitor,),
                    retain_response_logprobs=True,
                )
                scores, probabilities = probe_outputs(probe, result.captures[monitor])
                nll = response_nll(result)
                for index, record in enumerate(examples):
                    append_row(
                        path,
                        completed,
                        fields,
                        {
                            **common_metadata(
                                record,
                                plan_hash=plan_hash,
                                component_hash=freeze["component_set_sha256"],
                                implementation_commit=commit,
                            ),
                            "record_type": "prompt_confound",
                            "condition_id": condition_id,
                            "prompt_sha256": sha256_bytes(prompts[index].encode()),
                            "response_ids_sha256": response_hash(
                                response_ids[index], response_mask[index]
                            ),
                            "response_token_count": int(response_mask[index].sum()),
                            "probe_score": float(scores[index]),
                            **pool_fields(probabilities, response_mask, index),
                            "response_nll": float(nll[index]),
                        },
                    )
                del result
                release_memory()
            print(
                f"prompt confounds {concept}/label_{label}: batch {batch_index}; "
                f"rows {len(completed)}/448",
                flush=True,
            )
        print(f"prompt cell {cell_index}/4 complete", flush=True)
    if len(completed) != 448:
        raise RuntimeError(f"prompt confounds have {len(completed)} rows; expected 448")
    seal_jsonl(path, PROMPT_OUTPUT, completed, ("concept", "label", "example_id", "condition_id"))
    print(f"Sealed 448 prompt rows: {sha256_file(PROMPT_OUTPUT)}", flush=True)


def run_zero_ablation(
    runner: PairedInterventionRunner,
    records: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    freeze: dict[str, Any],
    analysis_plan: dict[str, Any],
    batch_size: int,
) -> None:
    path = ZERO_WORKING
    fields = ("example_id", "condition_id")
    completed = read_completed(path, fields)
    subset = deterministic_subset(records, "day13-confounds", 16)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in subset:
        cells[(record["concept"], int(record["label"]))].append(record)
    selected_ids = freeze["selected_candidates"]
    selected_sites = tuple(CANDIDATE_BY_ID[candidate_id].site for candidate_id in selected_ids)
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    plan_hash = sha256_file(PLAN_PATH)
    commit = git_head()
    for cell_index, key in enumerate(sorted(cells), start=1):
        concept, label = key
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        values = sorted(cells[key], key=lambda row: row["example_id"])
        for batch_index, examples in enumerate(batched(values, batch_size), start=1):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            triggered = runner.run(
                pair.triggered,
                capture_sites=(*selected_sites, monitor),
                retain_response_logits=True,
            )
            zero_captures = {
                site: CapturedActivation(
                    values=torch.zeros_like(triggered.captures[site].values),
                    response_ids=triggered.captures[site].response_ids.clone(),
                    response_mask=triggered.captures[site].response_mask.clone(),
                )
                for site in selected_sites
            }
            job = TransplantJob(
                "selected_k16_zero",
                tuple(
                    TransplantMember(site, zero_captures[site])
                    for site in selected_sites
                ),
            )
            result = vector.run_full_diagnostics(pair.triggered, (job,))
            kl = masked_token_kl(
                triggered.response_logits,
                result.response_logits[0],
                pair.triggered.response_mask,
                device=runner.device,
            )
            for index, record in enumerate(examples):
                append_row(
                    path,
                    completed,
                    fields,
                    {
                        **common_metadata(
                            record,
                            plan_hash=plan_hash,
                            component_hash=freeze["component_set_sha256"],
                            implementation_commit=commit,
                        ),
                        "record_type": "zero_ablation",
                        "condition_id": "selected_k16_zero_on_correct_trigger",
                        "group_id": "selected_k16",
                        "candidate_ids": selected_ids,
                        "source_condition": "zero",
                        "destination_condition": "correct_trigger",
                        "response_ids_sha256": response_hash(
                            pair.triggered.response_ids[index],
                            pair.triggered.response_mask[index],
                        ),
                        "response_token_count": int(
                            pair.triggered.response_mask[index].sum()
                        ),
                        "probe_score": float(result.probe_scores[0, index]),
                        **pool_fields(
                            result.probe_token_probabilities[0],
                            pair.triggered.response_mask,
                            index,
                        ),
                        "response_nll": float(result.response_nll[0, index]),
                        "response_kl": float(kl[index]),
                        "activation_rms_ratio": 0.0,
                        "kl_definition": "mean_valid_token_KL(destination||ablated)",
                    },
                )
            del triggered, result
            release_memory()
            print(
                f"zero ablation {concept}/label_{label}: batch {batch_index}; "
                f"rows {len(completed)}/64",
                flush=True,
            )
        print(f"zero cell {cell_index}/4 complete", flush=True)
    if len(completed) != 64:
        raise RuntimeError(f"zero ablation has {len(completed)} rows; expected 64")
    seal_jsonl(path, ZERO_OUTPUT, completed, ("concept", "label", "example_id"))
    print(f"Sealed 64 zero-ablation rows: {sha256_file(ZERO_OUTPUT)}", flush=True)


def trim_generated(ids: torch.Tensor, tokenizer: Any) -> list[int]:
    result = []
    for token in ids.tolist():
        if token == tokenizer.eos_token_id:
            break
        if token != tokenizer.pad_token_id:
            result.append(int(token))
    return result


def repeated_fourgram_fraction(ids: Sequence[int]) -> float:
    if len(ids) < 4:
        return 0.0
    fourgrams = [tuple(ids[index : index + 4]) for index in range(len(ids) - 3)]
    return 1.0 - len(set(fourgrams)) / len(fourgrams)


def token_f1(generated: Sequence[int], reference: Sequence[int]) -> float:
    if not generated or not reference:
        return 0.0
    overlap = sum((Counter(generated) & Counter(reference)).values())
    precision = overlap / len(generated)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def prefix_fraction(generated: Sequence[int], reference: Sequence[int]) -> float:
    if not reference:
        return 0.0
    common = 0
    for generated_token, reference_token in zip(generated, reference):
        if generated_token != reference_token:
            break
        common += 1
    return common / len(reference)


def pad_generated(
    sequences: Sequence[Sequence[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence)
        mask[index, : len(sequence)] = True
    return ids, mask


def run_generation(
    runner: PairedInterventionRunner,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    freeze: dict[str, Any],
    analysis_plan: dict[str, Any],
    batch_size: int,
) -> None:
    path = GENERATION_WORKING
    fields = ("example_id", "condition_id")
    completed = read_completed(path, fields)
    subset = deterministic_subset(records, "day13-generation", 8)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in subset:
        cells[(record["concept"], int(record["label"]))].append(record)
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    plan_hash = sha256_file(PLAN_PATH)
    commit = git_head()
    for cell_index, key in enumerate(sorted(cells), start=1):
        concept, label = key
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        values = sorted(cells[key], key=lambda row: row["example_id"])
        for batch_index, examples in enumerate(batched(values, batch_size), start=1):
            reference_ids, reference_mask = runner._tokenize_responses_once(
                [row["response"] for row in examples]
            )
            for condition_id in GENERATION_CONDITIONS:
                prompts = [prompt_for(condition_id, row, trigger, plan) for row in examples]
                rendered = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in prompts
                ]
                previous_padding = tokenizer.padding_side
                tokenizer.padding_side = "left"
                encoded = tokenizer(
                    rendered,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                tokenizer.padding_side = previous_padding
                with torch.inference_mode():
                    generated = runner.model.generate(
                        input_ids=encoded["input_ids"].to(runner.device),
                        attention_mask=encoded["attention_mask"].to(runner.device),
                        max_new_tokens=64,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                new_ids = generated[:, encoded["input_ids"].shape[1] :].detach().cpu()
                sequences = [trim_generated(row, tokenizer) for row in new_ids]
                nonempty_indices = [index for index, sequence in enumerate(sequences) if sequence]
                generated_scores: dict[int, float] = {}
                if nonempty_indices:
                    ids, mask = pad_generated(
                        [sequences[index] for index in nonempty_indices],
                        tokenizer.pad_token_id,
                    )
                    generated_condition = runner._prepare_condition(
                        condition_id,
                        [prompts[index] for index in nonempty_indices],
                        ids,
                        mask,
                    )
                    diagnostic = runner.run(
                        generated_condition, capture_sites=(monitor,)
                    )
                    scores, _probabilities = probe_outputs(
                        probe, diagnostic.captures[monitor]
                    )
                    generated_scores = {
                        example_index: float(scores[score_index])
                        for score_index, example_index in enumerate(nonempty_indices)
                    }
                for index, record in enumerate(examples):
                    sequence = sequences[index]
                    reference = reference_ids[index][reference_mask[index]].tolist()
                    append_row(
                        path,
                        completed,
                        fields,
                        {
                            **common_metadata(
                                record,
                                plan_hash=plan_hash,
                                component_hash=freeze["component_set_sha256"],
                                implementation_commit=commit,
                            ),
                            "record_type": "generation_diagnostic",
                            "condition_id": condition_id,
                            "prompt_sha256": sha256_bytes(prompts[index].encode()),
                            "generated_ids_sha256": sha256_bytes(
                                torch.tensor(sequence, dtype=torch.long).numpy().tobytes()
                            ),
                            "nonempty": bool(sequence),
                            "generated_token_count": len(sequence),
                            "repeated_fourgram_fraction": repeated_fourgram_fraction(sequence),
                            "reference_token_f1": token_f1(sequence, reference),
                            "reference_prefix_fraction": prefix_fraction(sequence, reference),
                            "generated_probe_score": generated_scores.get(index),
                            "decoding": {
                                "do_sample": False,
                                "max_new_tokens": 64,
                                "temperature": None,
                                "top_p": None,
                            },
                            "task_fidelity_scope": "descriptive token overlap; not a semantic correctness judgment",
                        },
                    )
                del generated
                release_memory()
            print(
                f"generation {concept}/label_{label}: batch {batch_index}; "
                f"rows {len(completed)}/128",
                flush=True,
            )
        print(f"generation cell {cell_index}/4 complete", flush=True)
    if len(completed) != 128:
        raise RuntimeError(f"generation diagnostics have {len(completed)} rows; expected 128")
    seal_jsonl(path, GENERATION_OUTPUT, completed, ("concept", "label", "example_id", "condition_id"))
    print(f"Sealed 128 generation rows: {sha256_file(GENERATION_OUTPUT)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    plan, freeze, _seal, analysis_plan = load_inputs()
    records = load_experimental_split("safety-test")
    if len(records) != 324:
        raise ValueError("safety split record count changed")
    runner, tokenizer = load_model()
    if not args.skip_prompt:
        run_prompt_confounds(
            runner, records, plan, freeze, analysis_plan, args.batch_size
        )
    if not args.skip_zero:
        run_zero_ablation(
            runner, records, plan, freeze, analysis_plan, args.batch_size
        )
    if not args.skip_generation:
        run_generation(
            runner,
            tokenizer,
            records,
            plan,
            freeze,
            analysis_plan,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
