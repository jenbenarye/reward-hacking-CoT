#!/usr/bin/env python3
"""Evaluate agent runs: check test pass rates and detect cheating behavior."""

import asyncio
import sys
from impossiblebench.analysis import DataLoader, LLMJudge


async def main():
    # 1. Load evaluation results
    print("Loading evaluation results...")
    loader = DataLoader(n_workers=4)
    loader.load_folder("logs")
    df = loader.to_df(dump=True)  # dump=True to get full transcripts

    # 2. Check how many tests passed
    total_samples = len(df)
    passed_samples = df[df['passed'] == True]
    num_passed = len(passed_samples)
    pass_rate = (num_passed / total_samples * 100) if total_samples > 0 else 0

    print(f"\n{'='*60}")
    print(f"Test Results: {num_passed}/{total_samples} passed ({pass_rate:.1f}%)")
    print(f"{'='*60}\n")

    # 3. Run binary evaluation on ALL samples (cheating vs legit)
    print("Running binary evaluation (cheating vs legit) on all samples...")

    # Get transcripts (filter out NaN dumps)
    valid_mask = df['dump'].notna()
    filtered_df = df[valid_mask]
    transcript_dumps = filtered_df['dump'].tolist()

    if not transcript_dumps:
        print("No transcripts available for analysis.")
        return

    judge = LLMJudge(model="claude-opus-4-20250514")
    binary_results = await judge.batch_evaluate_binary(transcript_dumps)

    # Count results by pass/fail status
    cheating_count = sum(1 for r in binary_results if r['judgment'] == 'cheating')
    legit_count = sum(1 for r in binary_results if r['judgment'] == 'legit')

    # Cross-reference with test results (order matches since we used filtered_df)
    cheating_passed = 0
    cheating_failed = 0
    legit_passed = 0
    legit_failed = 0

    for i, r in enumerate(binary_results):
        sample_passed = filtered_df.iloc[i]['passed']
        if r['judgment'] == 'cheating':
            if sample_passed:
                cheating_passed += 1
            else:
                cheating_failed += 1
        elif r['judgment'] == 'legit':
            if sample_passed:
                legit_passed += 1
            else:
                legit_failed += 1

    print(f"\nBinary Evaluation Results:")
    print(f"  Cheating: {cheating_count} total")
    print(f"    - Passed tests: {cheating_passed}")
    print(f"    - Failed tests: {cheating_failed}")
    print(f"  Legit: {legit_count} total")
    print(f"    - Passed tests: {legit_passed}")
    print(f"    - Failed tests: {legit_failed}")
    print(f"  Unknown/Error: {len(binary_results) - cheating_count - legit_count}\n")

    # 4. Run type evaluation for cheating samples
    cheating_indices = [i for i, r in enumerate(binary_results) if r['judgment'] == 'cheating']

    if cheating_indices:
        print(f"Running type evaluation on {len(cheating_indices)} cheating samples...")
        cheating_dumps = [transcript_dumps[i] for i in cheating_indices]
        type_results = await judge.batch_evaluate_type(cheating_dumps)

        # Count cheating types
        type_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'unknown': 0}
        for r in type_results:
            judgment = r['judgment']
            if judgment in type_counts:
                type_counts[judgment] += 1
            else:
                type_counts['unknown'] += 1

        print(f"\nCheating Type Classification:")
        print(f"  A (Modify test cases): {type_counts['A']}")
        print(f"  B (Overload operators): {type_counts['B']}")
        print(f"  C (Record extra states): {type_counts['C']}")
        print(f"  D (Special-casing): {type_counts['D']}")
        print(f"  E (Other): {type_counts['E']}")
        print(f"  Unknown: {type_counts['unknown']}")
    else:
        print("No cheating samples found - skipping type evaluation.")

    print("\nEvaluation complete!")

    # Explicitly close httpx connections to prevent hanging
    # The AsyncAnthropic client uses httpx internally which keeps connections open
    if hasattr(judge, 'anthropic_client'):
        try:
            client = judge.anthropic_client
            # Try different ways to access the httpx client
            if hasattr(client, '_client') and hasattr(client._client, 'aclose'):
                await client._client.aclose()
            elif hasattr(client, 'close'):
                await client.close()
            # Also try accessing through _http_client
            elif hasattr(client, '_http_client') and hasattr(client._http_client, 'aclose'):
                await client._http_client.aclose()
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    # Force exit - httpx connections may not close properly
    sys.exit(0)
