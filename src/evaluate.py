import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from predict import predict


def normalize_actions(actions):
    """Sort actions for set comparison."""
    return sorted(
        [(a["target"], a["action"]) for a in actions],
        key=lambda x: (x[0], x[1]),
    )


def evaluate(test_path, model_dir="outputs/bert", max_length=64):
    with open(test_path, "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    total = len(test_cases)
    domain_correct = 0
    camera_correct = 0
    closure_correct = 0
    unknown_correct = 0
    semantic_correct = 0

    results = []

    for case in test_cases:
        text = case["text"]
        expected_domain = case["domain"]

        pred = predict(text, model_dir=model_dir, max_length=max_length)
        pred_domain = pred.get("domain", "")
        domain_ok = pred_domain == expected_domain

        semantic_ok = False

        if expected_domain == "camera":
            expected_label = case["label"]
            pred_label = pred.get("label", "")
            semantic_ok = domain_ok and pred_label == expected_label
            if domain_ok:
                camera_correct += 1
            if semantic_ok:
                camera_correct = camera_correct  # already counted

        elif expected_domain == "closure":
            expected_actions = normalize_actions(case.get("actions", []))
            pred_actions = normalize_actions(pred.get("actions", []))
            semantic_ok = domain_ok and pred_actions == expected_actions
            if domain_ok:
                closure_correct += 1

        elif expected_domain == "unknown":
            semantic_ok = domain_ok
            if domain_ok:
                unknown_correct += 1

        if domain_ok:
            domain_correct += 1
        if semantic_ok:
            semantic_correct += 1

        results.append({
            "text": text,
            "expected_domain": expected_domain,
            "pred_domain": pred_domain,
            "domain_ok": domain_ok,
            "semantic_ok": semantic_ok,
            "expected": case.get("label") or case.get("actions"),
            "predicted": pred.get("label") or pred.get("actions"),
        })

    # Per-domain stats
    domain_counts = {}
    domain_sem_ok = {}
    for r in results:
        d = r["expected_domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
        if r["semantic_ok"]:
            domain_sem_ok[d] = domain_sem_ok.get(d, 0) + 1

    print("=" * 70)
    print(f"Test set: {test_path}")
    print(f"Total samples: {total}")
    print("=" * 70)
    print(f"{'Metric':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 70)
    print(f"{'Domain Accuracy':<30} {domain_correct:<10} {total:<10} {domain_correct/total:.1%}")
    print(f"{'Semantic Accuracy (overall)':<30} {semantic_correct:<10} {total:<10} {semantic_correct/total:.1%}")
    print("-" * 70)
    for d in sorted(domain_counts.keys()):
        c = domain_sem_ok.get(d, 0)
        t = domain_counts[d]
        print(f"  {d:<28} {c:<10} {t:<10} {c/t:.1%}")
    print("=" * 70)

    # Show failures
    failures = [r for r in results if not r["semantic_ok"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        print("-" * 70)
        for r in failures:
            status = "DOMAIN WRONG" if not r["domain_ok"] else "ACTION WRONG"
            print(f"  [{status}] {r['text']}")
            print(f"    expected: {r['expected']}")
            print(f"    predicted: {r['predicted']}")
    else:
        print("\nAll tests passed!")

    print(f"\nDomain accuracy: {domain_correct}/{total} = {domain_correct/total:.1%}")
    print(f"Semantic accuracy: {semantic_correct}/{total} = {semantic_correct/total:.1%}")

    return semantic_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/test_set.jsonl")
    parser.add_argument("--model_dir", default="outputs/bert")
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    evaluate(args.test, model_dir=args.model_dir, max_length=args.max_length)


if __name__ == "__main__":
    main()
