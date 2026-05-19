import json
import random
import requests
import typing as t

from datasets import load_dataset


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_JUDGE_MODEL = "gemma4:e4b"
OUTPUT_FILE_NAME = "counterfact_new_sample.json"


def ollama_inference(prompt: str, model = DEFAULT_JUDGE_MODEL, temperature: float = 0, max_tokens: int = 1024):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()['response']


def build_generation_prompt(entry: t.Dict[str, t.Any], num_prompts = 3):
    rewrite = entry['requested_rewrite']

    subject = rewrite["subject"]
    relation_id = rewrite["relation_id"]
    original_prompt = rewrite["prompt"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]

    num_medium = num_prompts // 2
    num_hard = num_prompts - num_medium

    return f"""
        Generate valid JSON only.

        We are building third-party cloze prompts for factual model editing.

        Edit fact:
        Subject: {subject}
        Relation ID: {relation_id}
        Template: {original_prompt}
        Old answer: {target_true}
        New answer: {target_new}

        After editing, the model should believe:
        {original_prompt.format(subject)} {target_new}

        Generate exactly {num_prompts} short cloze prompts.

        Goal:
        Each prompt should be centered on "{target_new}", but the expected answer should be "{subject}".

        The direction being tested is:
        "{target_new}" context -> "{subject}"

        Do not put "{subject}" inside the prompt.

        Make prompts easy for GPT-2 XL:
        - Use short cloze-style prompts.
        - Do not ask broad open-ended questions.
        - Prefer prompts ending in "is", "include", "includes", or "such as".
        - Avoid long explanations.

        Good prompt examples:
        "A famous landmark in Rome is"
        "A French actress whose mother tongue is English is"
        "One film director from Italy is"

        Candidate difficulty:
        Generate exactly {num_medium} prompts with difficulty "medium_same_type".
        Generate exactly {num_hard} prompts with difficulty "hard_same_context".

        For "medium_same_type":
        - Candidate 1 must be exactly "{subject}".
        - Candidates 2-4 must be plausible entities of the same semantic type as "{subject}".
        - Candidates 2-4 should NOT be strongly associated with "{target_new}".
        - Example: if the expected answer is "Eiffel Tower" and the context is "Rome", medium distractors could be "Statue of Liberty", "Big Ben", "Burj Khalifa".

        For "hard_same_context":
        - Candidate 1 must be exactly "{subject}".
        - Candidates 2-4 must be plausible entities of the same semantic type as "{subject}".
        - Candidates 2-4 should be strongly associated with "{target_new}" or naturally valid in the "{target_new}" context.
        - Example: if the expected answer is "Eiffel Tower" and the context is "Rome", hard distractors could be "Colosseum", "Trevi Fountain", "Pantheon".

        Return exactly this JSON structure:

        {{
        "third_party_prompts": [
            {{
            "prompt": "short cloze prompt",
            "expected_answer": "{subject}",
            "target_context": "{target_new}",
            "difficulty": "medium_same_type",
            "candidates": [
                "{subject}",
                "candidate 2",
                "candidate 3",
                "candidate 4"
            ]
            }}
        ]
        }}

        Rules:
        - Return valid JSON only.
        - No markdown.
        - No comments.
        - No trailing commas.
        - Use double quotes only.
        - Generate exactly {num_prompts} items in third_party_prompts.
        - Each candidates list must contain exactly 4 strings.
        - The first candidate must be exactly "{subject}".
        - The expected_answer must be exactly "{subject}".
        - The target_context must be exactly "{target_new}".
        - The difficulty must be either "medium_same_type" or "hard_same_context".
        - Generate exactly {num_medium} items with difficulty "medium_same_type".
        - Generate exactly {num_hard} items with difficulty "hard_same_context".
        - Do not include "{subject}" in the prompt text.
        - Do not duplicate prompts.
        - Do not duplicate candidates within the same candidates list.
    """.strip()


def extract_json(raw_text: str):
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Invalid response:\n{raw_text}")

    json_str = raw_text[start:end + 1]
    return json.loads(json_str)


def generate_third_party_prompts(entry: t.Dict[str, t.Any], model = DEFAULT_JUDGE_MODEL, num_prompts = 3):
    prompt = build_generation_prompt(entry, num_prompts=num_prompts)
    raw_response = ollama_inference(prompt, model=model)

    parsed_json = extract_json(raw_response)

    if "third_party_prompts" not in parsed_json:
        raise ValueError(f"Missing 'third_party_prompts' key in response:\n{parsed_json}")

    third_party_prompts = parsed_json["third_party_prompts"]

    if not isinstance(third_party_prompts, list):
        raise ValueError("third_party_prompts should be a list.")

    return third_party_prompts


if __name__ == "__main__":
    dataset = load_dataset('azhx/counterfact')
    dataset = dataset['train']

    random.seed(120)
    N_samples = 1000
    N_prompts_per_sample = 5

    sampled_indices = random.sample(range(len(dataset)), N_samples)
    results = []

    for i, idx in enumerate(sampled_indices):
        sample = dataset[idx]
        sample = dict(sample)

        subject = sample['requested_rewrite']['subject']
        target_new = sample['requested_rewrite']['target_new']['str']
        target_old = sample['requested_rewrite']['target_true']['str']
        print(f"Sample: {i}/{len(sampled_indices)} rewrite: {subject}, {target_old} -> {target_new}")

        try:
            third_party_prompts = generate_third_party_prompts(sample, model=DEFAULT_JUDGE_MODEL, num_prompts=N_prompts_per_sample)

            sample['third_party_prompts'] = third_party_prompts
            sample['third_party_generation_status'] = 'success'
        except Exception as e:
            print(f"Error: {e}")
            sample['third_party_prompts'] = []
            sample['third_party_generation_status'] = 'fail'
            sample['third_party_generation_error'] = str(e)

        results.append(sample)

    with open(OUTPUT_FILE_NAME, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)