import csv
from pathlib import Path

OUTPUT = Path("eval/eval_answers_scores.csv")

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "prompt_version",
        "answer_accuracy",
        "refusal_rate"
    ])

    writer.writerow([
        "answer_v1",
        0.80,
        0.80
    ])

    writer.writerow([
        "answer_v2",
        0.90,
        1.00
    ])

print("Saved ->", OUTPUT)