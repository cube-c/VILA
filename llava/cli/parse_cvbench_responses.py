"""Parse the response column from attention_ratios_cvbench.csv into a clean choice letter.

Handles formats:
  - "(A)"                       -> A
  - "(A) 3"                     -> A
  - "(A) left"                  -> A
  - "(left"                     -> None (no valid letter choice)
  - "above"                     -> None
  - "(The answer is D)"         -> D
  - "(The answer is D) 2)"      -> D
  - "(The image contains ... The answer is D.)" -> D

Adds columns:
  - predicted: extracted letter (A-F) or empty string if unparseable
  - correct: whether predicted == answer
"""

import argparse
import csv
import re


def extract_choice(response: str) -> str:
    """Extract a single choice letter (A-F) from a free-form response string."""
    # 1. Direct pattern: starts with "(X)" or "(X) ..."
    m = re.match(r"^\(([A-F])\)", response)
    if m:
        return m.group(1)

    # 2. "The answer is X" pattern
    m = re.search(r"[Tt]he answer is ([A-F])", response)
    if m:
        return m.group(1)

    # 3. Fallback: any (X) anywhere in the string
    m = re.search(r"\(([A-F])\)", response)
    if m:
        return m.group(1)

    return ""


def main():
    parser = argparse.ArgumentParser(description="Parse CV-Bench responses into choice letters")
    parser.add_argument("--input", "-i", default="output/attention_ratios_cvbench.csv")
    parser.add_argument("--output", "-o", default="output/attention_ratios_cvbench_parsed.csv")
    args = parser.parse_args()

    with open(args.input, newline="") as fin:
        reader = csv.DictReader(fin)
        in_fields = reader.fieldnames
        out_fields = list(in_fields) + ["predicted", "correct"]

        rows = []
        total = 0
        parsed = 0
        correct = 0
        for row in reader:
            pred = extract_choice(row["response"])
            row["predicted"] = pred
            row["correct"] = str(pred == row["answer"].strip("() ")) if pred else ""
            rows.append(row)

            total += 1
            if pred:
                parsed += 1
                if row["correct"] == "True":
                    correct += 1

    with open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    n_examples = total // 28 if total else 0  # 28 layers per example
    print(f"Rows:      {total}")
    print(f"Parsed:    {parsed}/{total} ({100*parsed/total:.1f}%)")
    print(f"Unparsed:  {total - parsed}")
    print(f"Correct:   {correct}/{parsed} ({100*correct/parsed:.1f}%)" if parsed else "Correct: N/A")
    print(f"Saved to:  {args.output}")


if __name__ == "__main__":
    main()
