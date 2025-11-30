#!/usr/bin/env python3
"""
Unified evaluation script for ImpossibleBench results.

Shows:
- Pass rates by dataset/variant
- Cheating vs legit classification by dataset
- Cheating type distribution by dataset
"""

import argparse
import asyncio
import pandas as pd
from impossiblebench.analysis import DataLoader, LLMJudge


async def evaluate_results(log_folder: str, llm_model: str, skip_llm_judge: bool, cheating_variants: list[str]):
    """Unified evaluation with pass rates and cheating analysis."""

    # Load evaluation results
    print(f"Loading evaluation results from {log_folder}...")
    loader = DataLoader(n_workers=4)
    loader.load_folder(log_folder)

    # Load with dumps if using LLM judge
    df = loader.to_df(dump=not skip_llm_judge)

    if df.empty:
        print(f"No evaluation results found in {log_folder}")
        return

    print(f"\nLoaded {len(df)} samples")

    # =========================================================================
    # SECTION 1: PASS RATES BY VARIANT
    # =========================================================================
    print(f"\n{'='*70}")
    print("PASS RATES BY VARIANT")
    print(f"{'='*70}\n")

    if 'variant' in df.columns:
        pass_rate_data = []
        for variant in sorted(df['variant'].unique()):
            variant_df = df[df['variant'] == variant]
            total = len(variant_df)
            passed = variant_df['passed'].sum()
            pass_rate = variant_df['passed'].mean()

            pass_rate_data.append({
                'Variant': variant,
                'Total': total,
                'Passed': passed,
                'Failed': total - passed,
                'Pass Rate': f"{pass_rate:.1%}",
            })

        pass_rate_df = pd.DataFrame(pass_rate_data)
        print(pass_rate_df.to_string(index=False))
    else:
        print("No variant information available")

    # =========================================================================
    # SECTION 2: LLM JUDGE ANALYSIS (if not skipped)
    # =========================================================================
    if not skip_llm_judge:
        # Filter to samples with dumps
        samples_with_dumps = df[df['dump'].notna()]
        if len(samples_with_dumps) == 0:
            print("\nNo transcripts available for LLM judge analysis.")
            print("Skipping cheating detection.")
            return

        # Filter to only specified variants for cheating analysis
        if cheating_variants:
            samples_with_dumps = samples_with_dumps[samples_with_dumps['variant'].isin(cheating_variants)]
            if len(samples_with_dumps) == 0:
                print(f"\nNo samples found for variants: {cheating_variants}")
                print("Skipping cheating detection.")
                return

        print(f"\n{'='*70}")
        print("CHEATING ANALYSIS BY VARIANT")
        print(f"{'='*70}")
        if cheating_variants:
            print(f"(Analyzing only: {', '.join(cheating_variants)})")
        print()

        try:
            # Cost warning
            total_samples = len(samples_with_dumps)
            print(f"⚠️  WARNING: About to analyze {total_samples} samples with {llm_model}")
            print(f"⚠️  This will make {total_samples} API calls for binary classification")
            print(f"⚠️  + additional calls for type classification on cheating samples")
            print(f"⚠️  Estimated cost: ${total_samples * 0.5:.0f}-${total_samples * 2:.0f} (depending on transcript size)")
            print(f"\n💡 To reduce cost:")
            print(f"   - Use cheaper model: --llm-model claude-sonnet-4-20250514")
            print(f"   - Analyze only oneoff: (already default)")
            print()
            response = input("Continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Cancelled.")
                return

            # Create LLM judge
            judge = LLMJudge(model=llm_model)

            # Analyze each variant
            all_variant_results = {}

            for variant in sorted(samples_with_dumps['variant'].unique()):
                variant_df = samples_with_dumps[samples_with_dumps['variant'] == variant]
                variant_dumps = variant_df['dump'].tolist()

                print(f"\nAnalyzing {variant} ({len(variant_df)} samples)...", end=" ", flush=True)

                # Binary classification
                binary_results = await judge.batch_evaluate_binary(variant_dumps)

                # Count binary results
                cheating_count = sum(1 for r in binary_results if r['judgment'] == 'cheating')
                legit_count = sum(1 for r in binary_results if r['judgment'] == 'legit')

                print(f"✓")

                # Cross-reference with test results
                cheating_passed = sum(1 for i, r in enumerate(binary_results)
                                     if r['judgment'] == 'cheating' and variant_df.iloc[i]['passed'])
                cheating_failed = cheating_count - cheating_passed
                legit_passed = sum(1 for i, r in enumerate(binary_results)
                                  if r['judgment'] == 'legit' and variant_df.iloc[i]['passed'])
                legit_failed = legit_count - legit_passed

                # Type classification (only on cheating samples)
                cheating_indices = [i for i, r in enumerate(binary_results) if r['judgment'] == 'cheating']
                type_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

                if cheating_indices:
                    print(f"  Classifying {len(cheating_indices)} cheating samples by type...", end=" ", flush=True)
                    cheating_dumps = [variant_dumps[i] for i in cheating_indices]
                    type_results = await judge.batch_evaluate_type(cheating_dumps)

                    for r in type_results:
                        judgment = r['judgment']
                        if judgment in type_counts:
                            type_counts[judgment] += 1
                    print(f"✓")

                all_variant_results[variant] = {
                    'total': len(variant_df),
                    'cheating': cheating_count,
                    'legit': legit_count,
                    'cheating_rate': cheating_count / len(binary_results) if binary_results else 0,
                    'cheating_passed': cheating_passed,
                    'cheating_failed': cheating_failed,
                    'legit_passed': legit_passed,
                    'legit_failed': legit_failed,
                    'type_counts': type_counts,
                }

            # Display binary classification summary
            print(f"\n{'─'*70}")
            print("BINARY CLASSIFICATION (Cheating vs Legit)")
            print(f"{'─'*70}\n")

            binary_data = []
            for variant, results in all_variant_results.items():
                binary_data.append({
                    'Variant': variant,
                    'Total': results['total'],
                    'Cheating': results['cheating'],
                    'Cheating %': f"{results['cheating_rate']:.1%}",
                    'Legit': results['legit'],
                    'Legit %': f"{(1-results['cheating_rate']):.1%}",
                })

            binary_df = pd.DataFrame(binary_data)
            print(binary_df.to_string(index=False))

            # Display cheating vs test results
            print(f"\n{'─'*70}")
            print("CHEATING vs TEST RESULTS")
            print(f"{'─'*70}\n")

            cheating_test_data = []
            for variant, results in all_variant_results.items():
                cheating_test_data.append({
                    'Variant': variant,
                    'Cheating & Passed': results['cheating_passed'],
                    'Cheating & Failed': results['cheating_failed'],
                    'Legit & Passed': results['legit_passed'],
                    'Legit & Failed': results['legit_failed'],
                })

            cheating_test_df = pd.DataFrame(cheating_test_data)
            print(cheating_test_df.to_string(index=False))

            # Display type classification
            print(f"\n{'─'*70}")
            print("CHEATING TYPE DISTRIBUTION")
            print(f"{'─'*70}\n")

            type_data = []
            for variant, results in all_variant_results.items():
                if results['cheating'] > 0:
                    tc = results['type_counts']
                    type_data.append({
                        'Variant': variant,
                        'Total Cheating': results['cheating'],
                        'A (Modify tests)': tc['A'],
                        'B (Overload ops)': tc['B'],
                        'C (Record state)': tc['C'],
                        'D (Special-case)': tc['D'],
                        'E (Other)': tc['E'],
                    })

            if type_data:
                type_df = pd.DataFrame(type_data)
                print(type_df.to_string(index=False))

                # Show percentages too
                print(f"\n{'─'*70}")
                print("CHEATING TYPE PERCENTAGES")
                print(f"{'─'*70}\n")

                for variant, results in all_variant_results.items():
                    if results['cheating'] > 0:
                        tc = results['type_counts']
                        total_cheating = results['cheating']
                        print(f"{variant}:")
                        print(f"  A (Modify test cases): {tc['A']:2d} ({tc['A']/total_cheating:5.1%})")
                        print(f"  B (Overload operators): {tc['B']:2d} ({tc['B']/total_cheating:5.1%})")
                        print(f"  C (Record extra states): {tc['C']:2d} ({tc['C']/total_cheating:5.1%})")
                        print(f"  D (Special-casing): {tc['D']:2d} ({tc['D']/total_cheating:5.1%})")
                        print(f"  E (Other): {tc['E']:2d} ({tc['E']/total_cheating:5.1%})")
            else:
                print("No cheating detected in any variant")

        except Exception as e:
            print(f"\nLLM Judge analysis failed: {e}")
            print("Make sure ANTHROPIC_API_KEY is set to run LLM judge analysis")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ImpossibleBench results with pass rates and cheating analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with cheating analysis (oneoff only, default)
  python evaluate_results.py logs/impossible_swebench

  # Just pass rates (no LLM judge)
  python evaluate_results.py logs/impossible_swebench --skip-llm-judge

  # Analyze all variants for cheating (expensive!)
  python evaluate_results.py logs/impossible_swebench --cheating-variants all

        """
    )

    parser.add_argument(
        "log_folder",
        default="logs/impossible_swebench",
        nargs="?",
        help="Path to log folder containing .eval files (default: logs/impossible_swebench)"
    )

    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM judge analysis (only show pass rates)"
    )

    parser.add_argument(
        "--llm-model",
        default="claude-opus-4-20250514",
        help="Model to use for LLM judge analysis (default: claude-opus-4-20250514)"
    )

    parser.add_argument(
        "--cheating-variants",
        nargs="+",
        default=["oneoff"],
        help="Which variants to analyze for cheating (default: oneoff only). Use 'all' for all variants."
    )

    args = parser.parse_args()

    # Handle 'all' special case
    cheating_variants = None if 'all' in args.cheating_variants else args.cheating_variants

    try:
        asyncio.run(evaluate_results(args.log_folder, args.llm_model, args.skip_llm_judge, cheating_variants))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
