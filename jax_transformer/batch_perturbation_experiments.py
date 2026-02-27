#!/usr/bin/env python
"""Batch perturbation experiments.

Starts from the baseline synonym lists and incrementally adds random words
one at a time. Runs training at checkpoints (every `skip` additions) so you
can watch how the histograms evolve as the perturbation grows.

The loop is N = 1, 1+skip, 1+2*skip, ... up to num_word_adds.

At each checkpoint, a training run produces histograms in its output folder.

Example usage:
    # Dry run to see what perturbations would happen
    PYTHONPATH=. python jax_transformer/batch_perturbation_experiments.py --dry_run

    # Run with defaults (100 adds, checkpoint every 10)
    PYTHONPATH=. python jax_transformer/batch_perturbation_experiments.py

    # Smaller experiment: 20 adds, checkpoint every 5
    PYTHONPATH=. python jax_transformer/batch_perturbation_experiments.py --num_word_adds 20 --skip 5
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

from experimental_code.datasets.anavan import make_cat_dog_worm_bird_anavan


def build_baseline_config():
    """Build the baseline synonym config (move=False case from weights.py)."""
    anavan = make_cat_dog_worm_bird_anavan()

    baseline = [
        # Left side (layer_width_idx = 0)
        {"name": "small",     "layer": 5, "position": 0, "layer_width_idx": 0,
         "token_list": sorted(anavan.get_synonyms_of_word("small"))},
        {"name": "dogs",      "layer": 4, "position": 1, "layer_width_idx": 0,
         "token_list": sorted(anavan.get_synonyms_of_word("dogs"))},
        {"name": "often",     "layer": 3, "position": 2, "layer_width_idx": 0,
         "token_list": sorted(anavan.get_synonyms_of_word("often"))},
        {"name": "fear",      "layer": 2, "position": 3, "layer_width_idx": 0,
         "token_list": sorted(anavan.get_synonyms_of_word("fear"))},
        {"name": "large",     "layer": 1, "position": 4, "layer_width_idx": 0,
         "token_list": sorted(anavan.get_synonyms_of_word("large"))},
        {"name": "cats",      "layer": 0, "position": 5, "layer_width_idx": 0,
         "token_list": sorted(anavan.get_synonyms_of_word("cats"))},

        # Right side (layer_width_idx = 1)
        {"name": "wriggly",   "layer": 5, "position": 0, "layer_width_idx": 1,
         "token_list": sorted(anavan.get_synonyms_of_word("wriggly"))},
        {"name": "worms",     "layer": 4, "position": 1, "layer_width_idx": 1,
         "token_list": sorted(anavan.get_synonyms_of_word("worms"))},
        {"name": "sometimes", "layer": 3, "position": 2, "layer_width_idx": 1,
         "token_list": sorted(anavan.get_synonyms_of_word("sometimes"))},
        {"name": "chase",     "layer": 2, "position": 3, "layer_width_idx": 1,
         "token_list": sorted(anavan.get_synonyms_of_word("chase"))},
        {"name": "angry",     "layer": 1, "position": 4, "layer_width_idx": 1,
         "token_list": sorted(anavan.get_synonyms_of_word("angry"))},
        {"name": "birds",     "layer": 0, "position": 5, "layer_width_idx": 1,
         "token_list": sorted(anavan.get_synonyms_of_word("birds"))},
    ]
    return baseline


def get_all_vocab_words():
    """Get all vocabulary words from the grammar."""
    anavan = make_cat_dog_worm_bird_anavan()
    all_words = set()
    for word_list in anavan.get_valid_left_ordered_words():
        all_words.update(word_list)
    for word_list in anavan.get_valid_right_ordered_words():
        all_words.update(word_list)
    return sorted(all_words)


def find_slot(config, layer_width_idx, position):
    """Find the config entry matching the given column and position."""
    for entry in config:
        if entry["layer_width_idx"] == layer_width_idx and entry["position"] == position:
            return entry
    return None


def add_random_word(config, vocab_words, rng, num_layers=6, max_attempts=100):
    """Try to add one random word to a random slot (positions 1-5, skip 0).

    Returns (success, message).
    """
    for _ in range(max_attempts):
        column = rng.randint(0, 1)
        position = rng.randint(1, num_layers - 1)  # 1 to 5 inclusive (skip position 0)
        word = rng.choice(vocab_words)

        slot = find_slot(config, column, position)
        if slot is None:
            continue

        if word not in slot["token_list"]:
            slot["token_list"].append(word)
            side = "L" if column == 0 else "R"
            return True, f"Added '{word}' to [{side}] pos={position} ({slot['name']})"

    return False, "Could not find a new word/slot combination"


def print_config(config):
    """Pretty-print the current config state."""
    for entry in config:
        side = "L" if entry["layer_width_idx"] == 0 else "R"
        print(f"  [{side}] pos={entry['position']} layer={entry['layer']} "
              f"{entry['name']:12s} ({len(entry['token_list']):2d} words): {entry['token_list']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental perturbation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Perturbation control
    parser.add_argument("--num_word_adds", type=int, default=100,
                        help="Total number of words to add (default: 100)")
    parser.add_argument("--skip", type=int, default=10,
                        help="Run training every N additions (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print perturbations, don't run training")

    # Training args (passed through to train.py)
    parser.add_argument("--training_text", type=str,
                        default="training_data/48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt")
    parser.add_argument("--prompt_text", type=str,
                        default="inference_data/inference_text.txt")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layer_width", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=200)

    return parser.parse_args()


def main():
    args = parse_args()

    config = build_baseline_config()
    vocab_words = get_all_vocab_words()
    rng = random.Random(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = (
        f"batch_perturb_{timestamp}_seed_{args.seed}"
        f"_adds_{args.num_word_adds}_skip_{args.skip}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    # Save baseline config
    baseline_path = os.path.join(experiment_dir, "config_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(config, f, indent=2)

    # Checkpoints: 1, 1+skip, 1+2*skip, ... (MATLAB 1:skip:num_word_adds)
    checkpoints = list(range(1, args.num_word_adds + 1, args.skip))
    if args.num_word_adds not in checkpoints:
        checkpoints.append(args.num_word_adds)

    print(f"Vocab: {len(vocab_words)} words")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Baseline saved to: {baseline_path}")
    print(f"Will add {args.num_word_adds} words total, "
          f"training at checkpoints: {checkpoints}")
    print(f"\nBaseline config:")
    print_config(config)
    print()

    total_added = 0
    results = []

    for checkpoint in checkpoints:
        # Add words one at a time until we reach this checkpoint
        while total_added < checkpoint:
            added, msg = add_random_word(
                config, vocab_words, rng, num_layers=args.num_layers
            )
            total_added += 1
            if added:
                print(f"  [{total_added:3d}] {msg}")
            else:
                print(f"  [{total_added:3d}] WARNING: {msg}")

        # Save config at this checkpoint
        config_path = os.path.join(
            experiment_dir, f"config_adds_{checkpoint:04d}.json"
        )
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*80}")
        print(f"CHECKPOINT: {checkpoint} words added | Config: {config_path}")
        print(f"{'='*80}")
        print_config(config)

        if args.dry_run:
            print()
            continue

        # Run training
        cmd = [
            sys.executable, "jax_transformer/train.py",
            args.training_text,
            "--prompt_text", args.prompt_text,
            "--num_layers", str(args.num_layers),
            "--layer_width", str(args.layer_width),
            "--num_samples", str(args.num_samples),
            "--num_epochs", str(args.num_epochs),
            "--silence_print",
            "--initialize_weights",
            "--synonym_config", config_path,
        ]
        print(f"\nRunning: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            env={**os.environ, "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Extract final loss
        final_loss = None
        for line in reversed(result.stdout.splitlines()):
            if "loss:" in line.lower():
                try:
                    final_loss = float(
                        line.split("loss:")[-1].strip().split()[0]
                    )
                except (ValueError, IndexError):
                    pass
                if final_loss is not None:
                    break

        results.append({
            "checkpoint": checkpoint,
            "num_words_added": checkpoint,
            "config_path": config_path,
            "final_loss": final_loss,
            "returncode": result.returncode,
        })

    # Save summary
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    if not args.dry_run and results:
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        for r in results:
            loss_str = (f"{r['final_loss']:.6e}"
                        if r['final_loss'] is not None else "N/A")
            status = ("OK" if r['returncode'] == 0
                      else f"FAIL({r['returncode']})")
            print(f"  N={r['checkpoint']:4d}: loss={loss_str}  status={status}")
        print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
