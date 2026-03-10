import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa-json", type=str, required=True,
                        help="Path to VQA JSON with bounding boxes")
    parser.add_argument("--output", "-o", type=str, default="cell_class.json")
    args = parser.parse_args()

    with open(args.vqa_json) as f:
        data = json.load(f)

    # Build per-cell classification: (phA, phB) -> consistent/counter/equal/mixed
    cell_class = {}
    for entry in data:
        img = entry["image"]
        m = re.search(r"phA(\d+)_phB(\d+)\.png$", img)
        if not m:
            continue
        cell = (int(m.group(1)), int(m.group(2)))
        obj1_cy = (entry["obj1"]["bbox"][1] + entry["obj1"]["bbox"][3]) / 2.0
        obj2_cy = (entry["obj2"]["bbox"][1] + entry["obj2"]["bbox"][3]) / 2.0
        if obj1_cy < obj2_cy:
            label = "consistent"
        elif obj1_cy > obj2_cy:
            label = "counter"
        else:
            label = "equal"
        if cell not in cell_class:
            cell_class[cell] = label
        elif cell_class[cell] != label:
            cell_class[cell] = "mixed"

    # (8,8) is consistent (equal + consistent -> consistent)
    if (8, 8) in cell_class:
        cell_class[(8, 8)] = "consistent"

    # Remove mixed cells
    mixed = [c for c, v in cell_class.items() if v == "mixed"]
    for c in mixed:
        del cell_class[c]
    print(f"Removed {len(mixed)} mixed cells")

    # Convert to serializable format
    out = {}
    for (phA, phB), label in sorted(cell_class.items()):
        out[f"{phA},{phB}"] = label

    counts = {}
    for v in cell_class.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"Cells: {counts}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
