"""Offline result verification: NumPy only, no source captures, salt or GPU."""
import argparse
import json
import os
from pathlib import Path
from report_v3 import load_results, validate_predictions


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'audit/recomputed_predictions.json')
    args = parser.parse_args()
    checks = validate_predictions(load_results(), args.output)
    print(json.dumps({'experiments': len(checks), 'passed': all(c['passed'] for c in checks),
                      'checks': ['macro-F1', 'accuracy', 'per-class P/R/F1/support', 'confusion matrix',
                                 'probabilities', 'five-fold coverage', 'split disjointness',
                                 'grouping', 'training cap', 'epoch selection']}))


if __name__ == '__main__':
    main()
