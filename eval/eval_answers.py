import csv
import json
from pathlib import Path
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qa import ask

QUESTIONS_FILE = Path("eval/qa_questions.json")
OUTPUT_FILE = Path("eval/eval_answers_scores.csv")

PROMPTS = [
    ("answer_v1", "prompts/answer_v1.txt"),
    ("answer_v2", "prompts/answer_v2.txt"),
]


def evaluate(prompt_name, prompt_file):

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    answerable_total = 0
    answerable_correct = 0

    refusal_total = 0
    refusal_correct = 0

    for q in questions:

        result = ask(
            q["question"],
            prompt_file=prompt_file
        )

        answer = result.answer.strip()

        if q["answerable"]:

            answerable_total += 1

            expected = q["expected_phrase"].strip().lower()

            if expected == "":
                if answer.lower() != "i don't know":
                    answerable_correct += 1
            else:
                if expected in answer.lower():
                    answerable_correct += 1

        else:

            refusal_total += 1

            if answer.lower() == "i don't know":
                refusal_correct += 1

    return (
        answerable_correct / answerable_total,
        refusal_correct / refusal_total
    )


def main():

    rows = []

    print("\nPrompt Evaluation\n")

    for name, prompt in PROMPTS:

        acc, refusal = evaluate(name, prompt)

        print(
            f"{name:12}"
            f" Accuracy={acc:.1%}"
            f" Refusal={refusal:.1%}"
        )

        rows.append([
            name,
            round(acc,3),
            round(refusal,3)
        ])

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "prompt_version",
            "answer_accuracy",
            "refusal_rate"
        ])

        writer.writerows(rows)

    print("\nSaved ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()