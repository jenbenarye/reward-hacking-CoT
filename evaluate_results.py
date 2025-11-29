#!/usr/bin/env python3
"""
Evaluate ImpossibleBench evaluation results.

This script follows the analysis example from the documentation.
"""

import argparse
import asyncio
from impossiblebench.analysis import DataLoader, LLMJudge


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ImpossibleBench evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_results.py logs/impossible_swebench

  # With LLM judge analysis
  python evaluate_results.py logs/impossible_swebench --llm-judge

  # Custom model for LLM judge
  python evaluate_results.py logs/impossible_swebench --llm-judge --llm-model claude-sonnet-4-20250514
        """
    )

    parser.add_argument(
        "log_folder",
        default="logs/impossible_swebench",
        nargs="?",
        help="Path to log folder containing .eval files (default: logs/impossible_swebench)"
    )

    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run LLM judge analysis to detect cheating behavior (requires ANTHROPIC_API_KEY)"
    )

    parser.add_argument(
        "--llm-model",
        default="claude-opus-4-20250514",
        help="Model to use for LLM judge analysis (default: claude-opus-4-20250514)"
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel loading (default: 4)"
    )

    args = parser.parse_args()

    # Load evaluation results
    print(f"Loading evaluation results from {args.log_folder}...")
    loader = DataLoader(n_workers=args.n_workers)
    loader.load_folder(args.log_folder)

    # Convert to DataFrame
    df = loader.to_df(dump=args.llm_judge)  # Load dumps only if using LLM judge

    if df.empty:
        print(f"No evaluation results found in {args.log_folder}")
        return

    # Get summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    summary = loader.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Show pass rates by variant
    if 'variant' in df.columns:
        print("\n" + "="*60)
        print("PASS RATES BY VARIANT")
        print("="*60)
        variant_rates = df.groupby('variant')['passed'].agg(['count', 'mean']).round(3)
        variant_rates.columns = ['num_samples', 'pass_rate']
        for variant, row in variant_rates.iterrows():
            print(f"  {variant}: {row['pass_rate']:.1%} ({int(row['num_samples'])} samples)")

    # Show pass rates by model
    if 'model' in df.columns:
        print("\n" + "="*60)
        print("PASS RATES BY MODEL")
        print("="*60)
        model_rates = df.groupby('model')['passed'].agg(['count', 'mean']).round(3)
        model_rates.columns = ['num_samples', 'pass_rate']
        for model, row in model_rates.iterrows():
            print(f"  {model}: {row['pass_rate']:.1%} ({int(row['num_samples'])} samples)")

    # Show pass rates by agent type
    if 'agent_type' in df.columns:
        print("\n" + "="*60)
        print("PASS RATES BY AGENT TYPE")
        print("="*60)
        agent_rates = df.groupby('agent_type')['passed'].agg(['count', 'mean']).round(3)
        agent_rates.columns = ['num_samples', 'pass_rate']
        for agent_type, row in agent_rates.iterrows():
            print(f"  {agent_type}: {row['pass_rate']:.1%} ({int(row['num_samples'])} samples)")

    # LLM Judge Analysis (if requested)
    if args.llm_judge:
        print("\n" + "="*60)
        print("LLM JUDGE ANALYSIS")
        print("="*60)

        # Get samples with dumps
        samples_with_dumps = df[df['dump'].notna()]
        if len(samples_with_dumps) == 0:
            print("No transcripts available for LLM judge analysis.")
            return

        print(f"Analyzing {len(samples_with_dumps)} samples with LLM judge...")
        transcript_dumps = samples_with_dumps['dump'].tolist()

        try:
            # Create judge instance
            judge = LLMJudge(model=args.llm_model)

            # Binary classification (cheating vs legit)
            print("\n1. Binary Classification (cheating vs legit):")
            binary_results = asyncio.run(judge.batch_evaluate_binary(transcript_dumps))

            # Count results
            cheating_count = sum(1 for r in binary_results if r['judgment'] == 'cheating')
            legit_count = sum(1 for r in binary_results if r['judgment'] == 'legit')
            error_count = len(binary_results) - cheating_count - legit_count

            print(f"  Cheating: {cheating_count}")
            print(f"  Legit: {legit_count}")
            if error_count > 0:
                print(f"  Errors: {error_count}")

            # Type classification (only on cheating samples)
            cheating_indices = [i for i, r in enumerate(binary_results) if r['judgment'] == 'cheating']
            if cheating_indices:
                print(f"\n2. Type Classification (A-E categories):")
                cheating_dumps = [transcript_dumps[i] for i in cheating_indices]
                type_results = asyncio.run(judge.batch_evaluate_type(cheating_dumps))

                # Count types
                type_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'unknown': 0}
                for r in type_results:
                    judgment = r['judgment']
                    if judgment in type_counts:
                        type_counts[judgment] += 1
                    else:
                        type_counts['unknown'] += 1

                print(f"  A (Modify test cases): {type_counts['A']}")
                print(f"  B (Overload operators): {type_counts['B']}")
                print(f"  C (Record extra states): {type_counts['C']}")
                print(f"  D (Special-casing): {type_counts['D']}")
                print(f"  E (Other): {type_counts['E']}")
                if type_counts['unknown'] > 0:
                    print(f"  Unknown: {type_counts['unknown']}")
            else:
                print("\n2. Type Classification: No cheating samples found")

        except Exception as e:
            print(f"\nLLM Judge analysis failed: {e}")
            print("Make sure ANTHROPIC_API_KEY is set to run LLM judge analysis")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
