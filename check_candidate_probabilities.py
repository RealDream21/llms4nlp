"""
Check whether a causal language model assigns the highest probability to the
correct answer among a small candidate set.

Example:

    uv run python check_candidate_probabilities.py --device mps

Custom prompt/candidates:

    uv run python check_candidate_probabilities.py \
        --prompt "One of the USA presidents was:" \
        --expected "Richard Nixon" \
        --candidates "Michael Jordan" "Michael Jackson" "Richard Nixon" "Pope"
"""

from __future__ import annotations

import argparse
import json
import math
import string
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score multiple answer candidates with a causal language model."
    )
    parser.add_argument("--model-name", default="openai-community/gpt2-xl")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--prompt", default="A notable author who writes in English is")
    parser.add_argument("--expected", default="J.K. Rowling")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["Michael Jordan", "Michael Jackson", "J.K. Rowling", "Stephen King"],
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of the readable table.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_arg)


def candidate_with_leading_space(prompt: str, candidate: str) -> str:
    if not candidate:
        return candidate
    if prompt and prompt[-1].isspace():
        return candidate
    if candidate[0].isspace() or candidate[0] in string.punctuation:
        return candidate
    return " " + candidate


def score_candidate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    candidate: str,
    device: torch.device,
) -> dict[str, Any]:
    """Score log P(candidate | prompt) with teacher forcing."""
    candidate_text = candidate_with_leading_space(prompt, candidate)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_ids)

    if not candidate_ids:
        return {
            "candidate": candidate,
            "candidate_text_scored": candidate_text,
            "tokens": [],
            "sum_logprob": float("-inf"),
            "avg_logprob_per_token": float("-inf"),
            "sequence_probability": 0.0,
            "num_tokens": 0,
        }

    input_ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    candidate_start = len(prompt_ids)

    token_scores = []
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]

        for token_pos in range(candidate_start, input_ids.shape[1]):
            shifted_pos = token_pos - 1
            token_id = labels[0, shifted_pos]
            token_logprob = float(log_probs[0, shifted_pos, token_id].item())
            token_scores.append(
                {
                    "token": tokenizer.convert_ids_to_tokens([int(token_id.item())])[0],
                    "logprob": token_logprob,
                    "probability": math.exp(token_logprob),
                }
            )

    sum_logprob = sum(item["logprob"] for item in token_scores)
    return {
        "candidate": candidate,
        "candidate_text_scored": candidate_text,
        "tokens": candidate_tokens,
        "token_scores": token_scores,
        "sum_logprob": sum_logprob,
        "avg_logprob_per_token": sum_logprob / len(candidate_ids),
        "sequence_probability": math.exp(sum_logprob),
        "num_tokens": len(candidate_ids),
    }


def add_candidate_normalized_probabilities(scored: list[dict[str, Any]]) -> None:
    max_logprob = max(item["sum_logprob"] for item in scored)
    denominator = sum(math.exp(item["sum_logprob"] - max_logprob) for item in scored)
    for item in scored:
        item["normalized_probability_among_candidates"] = (
            math.exp(item["sum_logprob"] - max_logprob) / denominator
        )


def rank_candidates(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    expected: str,
    candidates: list[str],
    device: torch.device,
) -> dict[str, Any]:
    if expected not in candidates:
        candidates = [*candidates, expected]

    scored = [
        score_candidate(model, tokenizer, prompt, candidate, device) for candidate in candidates
    ]
    add_candidate_normalized_probabilities(scored)

    ranked_by_sequence_probability = sorted(
        scored, key=lambda item: item["sum_logprob"], reverse=True
    )
    ranked_by_avg_token_probability = sorted(
        scored, key=lambda item: item["avg_logprob_per_token"], reverse=True
    )

    expected_rank_sequence = next(
        idx
        for idx, item in enumerate(ranked_by_sequence_probability, start=1)
        if item["candidate"] == expected
    )
    expected_rank_avg = next(
        idx
        for idx, item in enumerate(ranked_by_avg_token_probability, start=1)
        if item["candidate"] == expected
    )

    return {
        "prompt": prompt,
        "expected": expected,
        "ranked_by_sequence_probability": ranked_by_sequence_probability,
        "ranked_by_avg_token_probability": ranked_by_avg_token_probability,
        "expected_rank_sequence_probability": expected_rank_sequence,
        "expected_rank_avg_token_probability": expected_rank_avg,
        "correct_is_top_by_sequence_probability": expected_rank_sequence == 1,
        "correct_is_top_by_avg_token_probability": expected_rank_avg == 1,
    }


def print_table(result: dict[str, Any]) -> None:
    print(f"Prompt: {result['prompt']}")
    print(f"Expected answer: {result['expected']}")
    print()
    print("Ranked by full sequence probability P(candidate | prompt):")
    print(
        f"{'rank':>4}  {'candidate':<22}  {'tokens':>6}  "
        f"{'sum_logprob':>12}  {'seq_prob':>12}  {'norm_prob':>12}  {'avg_logprob':>12}"
    )
    for rank, item in enumerate(result["ranked_by_sequence_probability"], start=1):
        print(
            f"{rank:>4}  {item['candidate']:<22}  {item['num_tokens']:>6}  "
            f"{item['sum_logprob']:>12.4f}  {item['sequence_probability']:>12.4e}  "
            f"{item['normalized_probability_among_candidates']:>12.4f}  "
            f"{item['avg_logprob_per_token']:>12.4f}"
        )

    print()
    print(
        "Correct is top by sequence probability: "
        f"{result['correct_is_top_by_sequence_probability']}"
    )
    print(
        "Correct is top by average token log probability: "
        f"{result['correct_is_top_by_avg_token_probability']}"
    )
    print()
    print("Candidate tokenization:")
    for item in result["ranked_by_sequence_probability"]:
        print(f"- {item['candidate']}: {item['tokens']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Loading model={args.model_name} on device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    result = rank_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        expected=args.expected,
        candidates=args.candidates,
        device=device,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_table(result)


if __name__ == "__main__":
    main()
