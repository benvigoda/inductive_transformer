#!/usr/bin/env python
"""Sweep over random synonym list configurations.

Generates random assignments of vocab words to synonym list slots and runs
train.py with each configuration via --synonym_config.

Example usage:
    # Preview 5 random configs without running training
    PYTHONPATH=. python sweep_synonyms.py --num_runs 5 --dry_run

    # Run 3 configs with 1-4 words per slot
    PYTHONPATH=. python sweep_synonyms.py --num_runs 3 --min_words 1 --max_words 4

    # Run 10 configs with a specific seed for reproducibility
    PYTHONPATH=. python sweep_synonyms.py --num_runs 10 --seed 123
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

from experimental_code.datasets.anavan import make_cat_dog_worm_bird_anavan


def get_all_vocab_words():
    """Get all vocabulary words from the grammar."""
    anavan = make_cat_dog_worm_bird_anavan()
    all_words = set()
    for word_list in anavan.get_valid_left_ordered_words():
        all_words.update(word_list)
    for word_list in anavan.get_valid_right_ordered_words():
        all_words.update(word_list)
    return sorted(all_words)


def generate_random_config(num_layers, layer_width, vocab_words, rng,
                           min_words=1, max_words=None):
    """Generate a random synonym config.

    For each of the num_layers * layer_width slots, randomly samples
    between min_words and max_words from the full vocab. The layer/position
    pattern is fixed: layer goes (num_layers-1) down to 0, position goes
    0 up to (num_layers-1), for each layer_width_idx.
    """
    if max_words is None:
        max_words = len(vocab_words)
    max_words = min(max_words, len(vocab_words))

    config = []
    for layer_width_idx in range(layer_width):
        for position in range(num_layers):
            layer = num_layers - 1 - position
            num_words = rng.randint(min_words, max_words)
            words = rng.sample(vocab_words, num_words)
            config.append({
                "name": words[0],
                "layer": layer,
                "position": position,
                "layer_width_idx": layer_width_idx,
                "token_list": words,
            })
    return config


def print_config(config):
    """Pretty-print a synonym config."""
    for entry in config:
        side = "L" if entry["layer_width_idx"] == 0 else "R"
        print(f"  [{side}] layer={entry['layer']} pos={entry['position']} "
              f"name={entry['name']:12s} words={entry['token_list']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep over random synonym list configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Sweep control
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of random configurations to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the sweep RNG (controls which configs are generated)")
    parser.add_argument("--min_words", type=int, default=1,
                        help="Minimum words per synonym slot")
    parser.add_argument("--max_words", type=int, default=8,
                        help="Maximum words per synonym slot")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print configs, don't run training")

    # Training args (passed through to train.py)
    parser.add_argument("--training_text", type=str,
                        default="training_data/48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt")
    parser.add_argument("--prompt_text", type=str,
                        default="inference_data/inference_text.txt")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layer_width", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=200)

    return parser.parse_args()


def main():
    args = parse_args()

    vocab_words = get_all_vocab_words()
    print(f"Vocab ({len(vocab_words)} words): {vocab_words}")
    rng = random.Random(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = f"sweep_{timestamp}_seed_{args.seed}_runs_{args.num_runs}"
    os.makedirs(sweep_dir, exist_ok=True)

    results = []
    for run_idx in range(args.num_runs):
        config = generate_random_config(
            num_layers=args.num_layers,
            layer_width=args.layer_width,
            vocab_words=vocab_words,
            rng=rng,
            min_words=args.min_words,
            max_words=args.max_words,
        )

        config_path = os.path.join(sweep_dir, f"config_{run_idx:04d}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Run {run_idx + 1}/{args.num_runs} | Config: {config_path}")
        print(f"{'='*80}")
        print_config(config)

        if args.dry_run:
            continue

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
        print(f"Command: {' '.join(cmd)}")
        print()

        result = subprocess.run(
            cmd,
            env={**os.environ, "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )

        # Print stdout/stderr
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Try to extract final loss from output
        final_loss = None
        for line in reversed(result.stdout.splitlines()):
            if "loss:" in line.lower():
                try:
                    final_loss = float(line.split("loss:")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
                if final_loss is not None:
                    break

        results.append({
            "run_idx": run_idx,
            "config_path": config_path,
            "final_loss": final_loss,
            "returncode": result.returncode,
        })

    # Summary
    if not args.dry_run and results:
        summary_path = os.path.join(sweep_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print("SWEEP SUMMARY")
        print(f"{'='*80}")
        for r in results:
            loss_str = f"{r['final_loss']:.6e}" if r['final_loss'] is not None else "N/A"
            status = "OK" if r['returncode'] == 0 else f"FAIL({r['returncode']})"
            print(f"  Run {r['run_idx']:3d}: loss={loss_str}  status={status}  config={r['config_path']}")
        print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
