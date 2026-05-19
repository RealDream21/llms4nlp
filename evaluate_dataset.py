"""
Evaluate GPT-2 XL EasyEdit edits on a modified CounterFact dataset with
third-party/contextual consistency prompts.

Run from the repository root:

    uv run python evaluate_counterfact_3p_gpt2xl.py

Useful quick smoke test:

    uv run python evaluate_counterfact_3p_gpt2xl.py --max_samples 1 --device mps
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from easyeditor import BaseEditor, ROMEHyperParams
from tqdm import tqdm


HPARAMS_BY_METHOD = {
    "ROME-gp2xl": (ROMEHyperParams, "EasyEdit/hparams/ROME/gpt2-xl.yaml"),
    "ROME-llama3b": (ROMEHyperParams, "EasyEdit/hparams/ROME/llama3.2-3b.yaml")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CounterFact edits with third-party consistency prompts."
    )
    parser.add_argument("--data_path", default="counterfact_new_sample.json")
    parser.add_argument("--output_path", default="counterfact_3p_eval_results.json")
    parser.add_argument("--summary_path", default="counterfact_3p_eval_summary.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=sorted(HPARAMS_BY_METHOD),
        default="ROME-gp2xl",
        help="Editing method. Defaults to the same ROME setup as the GPT-2 XL example.",
    )
    parser.add_argument(
        "--hparams",
        default=None,
        help="EasyEdit hparams YAML. Defaults to the GPT-2 XL hparams for --method.",
    )
    parser.add_argument(
        "--model-name",
        default="openai-community/gpt2-xl",
        help="Override hparams.model_name, matching counterfact_gpt2xl_20_example.py.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="auto, cuda, cuda:0, mps, cpu, or an EasyEdit integer GPU id.",
    )
    parser.add_argument(
        "--reload_model_each_edit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload the base model after each sample instead of restoring edited weights.",
    )
    parser.add_argument(
        "--test-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to EasyEdit editor.edit. Disabled by default because it is slow.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str | int:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_arg.isdigit():
        return int(device_arg)
    if device_arg == "cuda":
        return "cuda:0"
    return device_arg


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def maybe_move_model_for_scoring(editor: BaseEditor, device: str | int) -> None:
    if isinstance(device, int):
        target = torch.device(f"cuda:{device}")
    else:
        target = torch.device(device)
    if model_device(editor.model) != target:
        editor.model.to(target)


def build_editor(args: argparse.Namespace) -> BaseEditor:
    hparams_cls, default_hparams = HPARAMS_BY_METHOD[args.method]
    hparams = hparams_cls.from_hparams(args.hparams or default_hparams)
    if args.model_name is not None:
        hparams.model_name = args.model_name
    hparams.device = resolve_device(args.device)

    print(f"Loading editor: method={args.method}, model={hparams.model_name}, device={hparams.device}")
    editor = BaseEditor.from_hparams(hparams)
    maybe_move_model_for_scoring(editor, hparams.device)
    editor.model.eval()
    return editor


def load_dataset(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list, got {type(data).__name__}")
    return data[:max_samples] if max_samples is not None else data


def candidate_with_leading_space(prompt: str, candidate: str) -> str:
    if not candidate:
        return candidate
    if prompt and prompt[-1].isspace():
        return candidate
    if candidate[0].isspace() or candidate[0] in string.punctuation:
        return candidate
    return " " + candidate


def score_candidate(model, tokenizer, prompt: str, candidate: str, device=None) -> dict[str, float | int]:
    """Score log P(candidate | prompt) with teacher forcing.

    Direct evaluation scores target_new given subject-centric prompts; third-party
    evaluation scores the edited subject given new-object/context-centered prompts.
    """
    if device is None:
        device = model_device(model)
    candidate_text = candidate_with_leading_space(prompt, candidate)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    if not candidate_ids:
        return {"sum_logprob": float("-inf"), "avg_logprob_per_token": float("-inf"), "num_tokens": 0}

    input_ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    candidate_start = len(prompt_ids)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]

        token_logprobs = []
        for token_pos in range(candidate_start, input_ids.shape[1]):
            shifted_pos = token_pos - 1
            token_id = labels[0, shifted_pos]
            token_logprobs.append(log_probs[0, shifted_pos, token_id])

    stacked = torch.stack(token_logprobs)
    sum_logprob = float(stacked.sum().item())
    return {
        "sum_logprob": sum_logprob,
        "avg_logprob_per_token": float((stacked.mean()).item()),
        "num_tokens": int(len(candidate_ids)),
    }


def evaluate_direct(model, tokenizer, rewrite: dict[str, Any]) -> dict[str, Any]:
    subject = rewrite["subject"]
    prompt = rewrite["prompt"].format(subject)
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite.get("target_true", {}).get("str")

    result = {
        "prompt": prompt,
        "target_new": target_new,
        "target_new_score": score_candidate(model, tokenizer, prompt, target_new),
    }
    if target_true is not None:
        result["target_true"] = target_true
        result["target_true_score"] = score_candidate(model, tokenizer, prompt, target_true)
    return result


def normalize_candidates(candidates: Any, expected_answer: str) -> list[str]:
    normalized = [str(item) for item in candidates] if isinstance(candidates, list) else []
    if expected_answer not in normalized:
        normalized.insert(0, expected_answer)
    return normalized


def rank_candidates(model, tokenizer, prompt: str, expected_answer: str, candidates: list[str]) -> dict[str, Any]:
    scored = []
    for candidate in candidates:
        scores = score_candidate(model, tokenizer, prompt, candidate)
        scored.append({"candidate": candidate, **scores})

    ranked = sorted(scored, key=lambda item: item["avg_logprob_per_token"], reverse=True)
    expected_rank = next(
        idx for idx, item in enumerate(ranked, start=1) if item["candidate"] == expected_answer
    )
    expected_item = ranked[expected_rank - 1]
    top_item = ranked[0]

    return {
        "ranked_candidates": ranked,
        "expected_answer_rank": expected_rank,
        "expected_answer_score": expected_item["avg_logprob_per_token"],
        "top_candidate": top_item["candidate"],
        "top_candidate_score": top_item["avg_logprob_per_token"],
        "accuracy_at_1": top_item["candidate"] == expected_answer,
        "reciprocal_rank": 1.0 / expected_rank,
    }


def evaluate_third_party(model, tokenizer, prompts: Any) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(prompts, list):
        return [], 1 if prompts is not None else 0

    results = []
    skipped = 0
    for item in prompts:
        if not isinstance(item, dict):
            skipped += 1
            continue
        prompt = item.get("prompt")
        expected_answer = item.get("expected_answer")
        if not prompt or not expected_answer:
            skipped += 1
            continue

        candidates = normalize_candidates(item.get("candidates"), str(expected_answer))
        ranking = rank_candidates(model, tokenizer, str(prompt), str(expected_answer), candidates)
        results.append(
            {
                "prompt": str(prompt),
                "expected_answer": str(expected_answer),
                "difficulty": str(item.get("difficulty", "unknown")),
                "target_context": item.get("target_context"),
                "candidates": candidates,
                **ranking,
            }
        )

    return results, skipped


def apply_edit(editor: BaseEditor, rewrite: dict[str, Any], test_generation: bool) -> dict[str, Any]:
    subject = rewrite["subject"]
    prompt = rewrite["prompt"].format(subject)
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite.get("target_true", {}).get("str", "<|endoftext|>")

    _metrics, edited_model, weights_copy = editor.edit(
        prompts=prompt,
        target_new=target_new,
        ground_truth=target_true,
        subject=subject,
        sequential_edit=False,
        test_generation=test_generation,
        verbose=False,
    )
    editor.model = edited_model
    editor.model.eval()
    return weights_copy


def restore_weights(model: torch.nn.Module, weights_copy: Any) -> bool:
    if callable(weights_copy):
        with torch.no_grad():
            weights_copy()
        return True
    if not isinstance(weights_copy, dict) or not weights_copy:
        return False

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, original_value in weights_copy.items():
            if name not in params:
                return False
            params[name].data.copy_(original_value.to(device=params[name].device, dtype=params[name].dtype))
    return True


def combine_third_party_results(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    combined = []
    for before_item, after_item in zip(before, after):
        before_rank = before_item["expected_answer_rank"]
        after_rank = after_item["expected_answer_rank"]
        before_score = before_item["expected_answer_score"]
        after_score = after_item["expected_answer_score"]
        rank_improvement = before_rank - after_rank
        score_gain = after_score - before_score
        combined.append(
            {
                "prompt": before_item["prompt"],
                "expected_answer": before_item["expected_answer"],
                "difficulty": before_item.get("difficulty", "unknown"),
                "target_context": before_item.get("target_context"),
                "candidates": before_item["candidates"],
                "before": before_item,
                "after": after_item,
                "rank_before": before_rank,
                "rank_after": after_rank,
                "reciprocal_rank_before": before_item["reciprocal_rank"],
                "reciprocal_rank_after": after_item["reciprocal_rank"],
                "is_top1_before": before_item["accuracy_at_1"],
                "is_top1_after": after_item["accuracy_at_1"],
                "rank_improvement": rank_improvement,
                "score_gain": score_gain,
                "expected_answer_avg_logprob_gain": score_gain,
            }
        )
    return combined


def compute_group_metrics(prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not prompt_results:
        return {
            "num_prompts": 0,
            "accuracy_at_1_before": None,
            "accuracy_at_1_after": None,
            "mrr_before": None,
            "mrr_after": None,
            "average_rank_improvement": None,
            "average_expected_answer_avg_logprob_gain": None,
        }

    before_acc = [float(item["before"]["accuracy_at_1"]) for item in prompt_results]
    after_acc = [float(item["after"]["accuracy_at_1"]) for item in prompt_results]
    before_mrr = [item["before"]["reciprocal_rank"] for item in prompt_results]
    after_mrr = [item["after"]["reciprocal_rank"] for item in prompt_results]
    rank_improvements = [item["rank_improvement"] for item in prompt_results]
    score_gains = [item["score_gain"] for item in prompt_results]

    return {
        "num_prompts": len(prompt_results),
        "accuracy_at_1_before": mean(before_acc),
        "accuracy_at_1_after": mean(after_acc),
        "mrr_before": mean(before_mrr),
        "mrr_after": mean(after_mrr),
        "average_rank_improvement": mean(rank_improvements),
        "average_expected_answer_avg_logprob_gain": mean(score_gains),
    }


def compute_difficulty_breakdown(prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "medium_same_type": [],
        "hard_same_context": [],
    }
    for item in prompt_results:
        difficulty = str(item.get("difficulty", "unknown"))
        grouped.setdefault(difficulty, []).append(item)
    return {
        difficulty: compute_group_metrics(group_results)
        for difficulty, group_results in grouped.items()
    }


def aggregate(results: list[dict[str, Any]], skipped_third_party: int) -> dict[str, Any]:
    direct_before = [
        item["direct_before"]["target_new_score"]["avg_logprob_per_token"] for item in results
    ]
    direct_after = [
        item["direct_after"]["target_new_score"]["avg_logprob_per_token"] for item in results
    ]
    third_party = [
        prompt_result
        for item in results
        for prompt_result in item["third_party_results"]
    ]

    third_party_metrics = compute_group_metrics(third_party)

    return {
        "num_samples": len(results),
        "num_third_party_prompts": len(third_party),
        "skipped_third_party_prompts": skipped_third_party,
        "direct_target_new_avg_score_before": mean(direct_before) if direct_before else None,
        "direct_target_new_avg_score_after": mean(direct_after) if direct_after else None,
        "direct_target_new_avg_score_gain": mean(
            [after - before for before, after in zip(direct_before, direct_after)]
        )
        if direct_before
        else None,
        "third_party_accuracy_at_1_before": third_party_metrics["accuracy_at_1_before"],
        "third_party_accuracy_at_1_after": third_party_metrics["accuracy_at_1_after"],
        "third_party_mrr_before": third_party_metrics["mrr_before"],
        "third_party_mrr_after": third_party_metrics["mrr_after"],
        "average_rank_improvement": third_party_metrics["average_rank_improvement"],
        "average_expected_answer_avg_logprob_gain": third_party_metrics[
            "average_expected_answer_avg_logprob_gain"
        ],
        "difficulty_breakdown": compute_difficulty_breakdown(third_party),
    }


def main() -> None:
    args = parse_args()
    samples = load_dataset(Path(args.data_path), args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.data_path}")

    editor = build_editor(args)
    results = []
    skipped_third_party = 0

    for sample in tqdm(samples, desc="Evaluating samples"):
        case_id = sample.get("case_id")
        rewrite = sample.get("requested_rewrite")
        if not isinstance(rewrite, dict):
            print(f"Skipping malformed sample case_id={case_id}: missing requested_rewrite")
            continue

        direct_before = evaluate_direct(editor.model, editor.tok, rewrite)
        third_before, skipped_before = evaluate_third_party(
            editor.model, editor.tok, sample.get("third_party_prompts")
        )

        weights_copy = apply_edit(editor, rewrite, args.test_generation)

        direct_after = evaluate_direct(editor.model, editor.tok, rewrite)
        third_after, skipped_after = evaluate_third_party(
            editor.model, editor.tok, sample.get("third_party_prompts")
        )
        skipped_third_party += max(skipped_before, skipped_after)

        results.append(
            {
                "case_id": case_id,
                "requested_rewrite": rewrite,
                "direct_before": direct_before,
                "direct_after": direct_after,
                "third_party_results": combine_third_party_results(third_before, third_after),
            }
        )

        if args.reload_model_each_edit:
            del editor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            editor = build_editor(args)
        elif not restore_weights(editor.model, weights_copy):
            raise RuntimeError(
                "Could not restore original weights from EasyEdit. "
                "Rerun with --reload_model_each_edit for the safe fallback."
            )

    summary = aggregate(results, skipped_third_party)

    with Path(args.output_path).open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with Path(args.summary_path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote detailed results to {args.output_path}")
    print(f"Wrote aggregate summary to {args.summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
