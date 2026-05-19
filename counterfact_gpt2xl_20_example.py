"""
Example: edit GPT-2 XL on 20 CounterFact examples with EasyEdit.

Run from the repository root:

    uv run python counterfact_gpt2xl_20_example.py

Useful variants:

    uv run python counterfact_gpt2xl_20_example.py \
        --dataset-name azhx/counterfact \
        --split train \
        --method ROME \
        --model-name gpt2-xl \
        --limit 20

Notes:
    - GPT-2 XL editing is large. A CUDA GPU or Apple MPS device is strongly
      recommended.
    - By default, the EasyEdit hparams file points to ./hugging_cache/gpt2-xl.
      Use --model-name gpt2-xl if you want Transformers to resolve it from the
      Hugging Face cache instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from datasets import load_dataset
from easyeditor import BaseEditor, MEMITHyperParams, ROMEHyperParams


HPARAMS_BY_METHOD = {
    "ROME": (ROMEHyperParams, "EasyEdit/hparams/ROME/gpt2-xl.yaml")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EasyEdit editing/evaluation on 20 CounterFact examples."
    )
    parser.add_argument(
        "--method",
        choices=sorted(HPARAMS_BY_METHOD),
        default="ROME",
        help="Editing method. ROME matches gpt-xl.ipynb and is lighter on MPS.",
    )
    parser.add_argument(
        "--dataset-name",
        default="azhx/counterfact",
        help="Hugging Face dataset name to load with datasets.load_dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use after load_dataset(...), for example train or test.",
    )
    parser.add_argument(
        "--hparams",
        default=None,
        help="EasyEdit hparams YAML. Defaults to the GPT-2 XL hparams for --method.",
    )
    parser.add_argument(
        "--model-name",
        default="openai-community/gpt2-xl",
        help="Override hparams.model_name, for example gpt2-xl or ./hugging_cache/gpt2-xl.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Override hparams.device, for example 0, cuda:0, cpu, or mps.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of examples to edit.")
    parser.add_argument(
        "--output-dir",
        default="outputs/counterfact_gpt2xl_20",
        help="Directory where metrics and summary JSON files are written.",
    )
    parser.add_argument(
        "--sequential-edit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply all edits to the same model before evaluating the edited model.",
    )
    parser.add_argument(
        "--test-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run EasyEdit fluency/generation evaluation. Slower.",
    )
    return parser.parse_args()


def load_counterfact_examples(dataset_name: str, split: str, limit: int) -> dict[str, list]:
    ds = load_dataset(dataset_name)
    if split not in ds:
        raise ValueError(f"Split {split!r} not found. Available splits: {list(ds.keys())}")

    split_ds = ds[split]
    examples = split_ds.select(range(min(limit, len(split_ds))))
    prompts = []
    subjects = []
    ground_truth = []
    target_new = []
    rephrase_prompts = []
    locality_prompts = []
    locality_answers = []

    for example in examples:
        rewrite = example["requested_rewrite"]
        subject = rewrite["subject"]

        prompts.append(rewrite["prompt"].format(subject))
        subjects.append(subject)
        ground_truth.append(rewrite["target_true"]["str"])
        target_new.append(rewrite["target_new"]["str"])
        rephrase_prompts.append(example["paraphrase_prompts"][0])

        neighborhood_prompts = example.get("neighborhood_prompts", [])
        locality_prompts.append(neighborhood_prompts)
        locality_answers.append([rewrite["target_true"]["str"]] * len(neighborhood_prompts))

    return {
        "prompts": prompts,
        "subjects": subjects,
        "ground_truth": ground_truth,
        "target_new": target_new,
        "rephrase_prompts": rephrase_prompts,
        "locality_inputs": {
            "neighborhood": {
                "prompt": locality_prompts,
                "ground_truth": locality_answers,
            }
        },
    }


def numeric_values(value) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def average_metric(metrics: list[dict], section: str, key: str) -> float | None:
    values = []
    for case in metrics:
        if key in case.get(section, {}):
            values.extend(numeric_values(case[section][key]))
    if not values:
        return None
    return float(mean(values))


def average_locality(metrics: list[dict]) -> float | None:
    values = []
    for case in metrics:
        locality = case.get("post", {}).get("locality", {})
        for key, value in locality.items():
            if key.endswith("_acc"):
                values.extend(value)
    return float(mean(values)) if values else None


def write_outputs(metrics: list[dict], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary = {
        "num_examples": limit,
        "pre_rewrite_acc": average_metric(metrics, "pre", "rewrite_acc"),
        "post_rewrite_acc": average_metric(metrics, "post", "rewrite_acc"),
        "post_rephrase_acc": average_metric(metrics, "post", "rephrase_acc"),
        "post_locality_acc": average_locality(metrics),
    }
    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote raw metrics to {metrics_file}")
    print(f"Wrote summary to {summary_file}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()

    data = load_counterfact_examples(args.dataset_name, args.split, args.limit)

    hparams_cls, default_hparams = HPARAMS_BY_METHOD[args.method]
    hparams = hparams_cls.from_hparams(args.hparams or default_hparams)
    if args.model_name is not None:
        hparams.model_name = args.model_name
    if args.device is not None:
        hparams.device = int(args.device) if args.device.isdigit() else args.device

    print(f"Loaded {len(data['prompts'])} CounterFact examples from {args.dataset_name}/{args.split}")
    print(f"Editing method: {args.method}")
    print(f"Editing model: {hparams.model_name}")
    print(f"Device: {hparams.device}")
    print(f"Sequential edit: {args.sequential_edit}")

    editor = BaseEditor.from_hparams(hparams)
    metrics, _edited_model, _weights_copy = editor.edit(
        prompts=data["prompts"],
        target_new=data["target_new"],
        ground_truth=data["ground_truth"],
        subject=data["subjects"],
        rephrase_prompts=data["rephrase_prompts"],
        locality_inputs=data["locality_inputs"],
        sequential_edit=args.sequential_edit,
        test_generation=args.test_generation,
    )

    write_outputs(metrics, Path(args.output_dir), len(data["prompts"]))


if __name__ == "__main__":
    main()
