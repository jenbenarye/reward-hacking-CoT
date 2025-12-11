#!/usr/bin/env python3
"""
Run the original ImpossibleBench judgment pipeline on a specific run.

This script:
1. Loads transcripts from a specific run using DataLoader
2. Runs binary evaluation (cheating vs legit)
3. Runs type evaluation on samples classified as cheating
4. Saves results to JSON files
"""

import argparse
import asyncio
import json
from pathlib import Path
from impossiblebench.analysis import DataLoader, LLMJudge


async def main():
    parser = argparse.ArgumentParser(
        description="Run original judgment pipeline on a specific run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on gpt5-monitoring-oneoff-100
  python run_original_judge.py gpt5-monitoring-oneoff-100

  # Use a different judge model
  python run_original_judge.py gpt5-monitoring-oneoff-100 --model claude-sonnet-4-20250514

  # Save to custom output directory
  python run_original_judge.py gpt5-monitoring-oneoff-100 --output-dir results/judgments
        """
    )
    
    parser.add_argument(
        "run_name",
        help="Name of the run to judge (e.g., gpt5-monitoring-oneoff-100)"
    )
    
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/impossible_swebench",
        help="Directory containing eval logs (default: logs/impossible_swebench)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-20250514",
        help="Judge model to use (default: claude-opus-4-20250514). "
             "Can use Gemini models like 'gemini/gemini-2.5-flash' or 'gemini/gemini-1.5-pro' if Anthropic credits are low."
    )
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=20,
        help="Maximum concurrent API requests (default: 20)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="judgments_original",
        help="Output directory for results (default: judgments_original)"
    )
    
    parser.add_argument(
        "--skip-binary",
        action="store_true",
        help="Skip binary evaluation (only run type evaluation if cheating samples provided)"
    )
    
    parser.add_argument(
        "--skip-type",
        action="store_true",
        help="Skip type evaluation"
    )
    
    parser.add_argument(
        "--system-prompt-override",
        type=str,
        default=None,
        help="System prompt to show the judge (instead of the one in the dump). "
             "Can be a file path (reads from file) or direct text."
    )
    
    args = parser.parse_args()
    
    # Set up paths
    logs_dir = Path(args.logs_dir)
    run_dir = logs_dir / args.run_name
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    print(f"Loading transcripts from: {run_dir}")
    
    # Load evaluation results
    loader = DataLoader(n_workers=4)
    loader.load_folder(str(run_dir), pattern="*.eval")
    
    # Get DataFrame with dumps
    df = loader.to_df(dump=True)
    
    if df.empty:
        print(f"Error: No evaluation results found in {run_dir}")
        return
    
    # Filter to samples with dumps
    samples_with_dumps = df[df['dump'].notna()]
    
    if len(samples_with_dumps) == 0:
        print(f"Error: No transcripts (dumps) found in {run_dir}")
        return
    
    print(f"\nFound {len(samples_with_dumps)} samples with transcripts")
    print(f"Total samples: {len(df)}")
    
    # Extract dumps
    dumps = samples_with_dumps['dump'].tolist()
    sample_metadata = samples_with_dumps[['task_name', 'sample_id', 'sample_original_id', 'passed', 'score']].to_dict('records')
    
    # Parse system prompt override if provided
    system_prompt_override = None
    if args.system_prompt_override:
        prompt_path = Path(args.system_prompt_override)
        if prompt_path.exists() and prompt_path.is_file():
            # Read from file
            with open(prompt_path, 'r') as f:
                system_prompt_override = f.read().strip()
            print(f"\nUsing system prompt override from file: {prompt_path}")
        else:
            # Use as direct text
            system_prompt_override = args.system_prompt_override
            print(f"\nUsing system prompt override (direct text)")
        print(f"System prompt override: {system_prompt_override[:100]}..." if len(system_prompt_override) > 100 else f"System prompt override: {system_prompt_override}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize judge
    print(f"\nInitializing LLM Judge with model: {args.model}")
    print(f"Note: If using Anthropic models, ensure ANTHROPIC_API_KEY is set and account has credits.")
    print(f"      If using Gemini models, ensure GOOGLE_API_KEY is set.")
    try:
        judge = LLMJudge(model=args.model, max_concurrency=args.max_concurrency)
    except Exception as e:
        print(f"Error initializing judge: {e}")
        if "anthropic" in args.model.lower() or "claude" in args.model.lower():
            print("\n💡 Tip: If you're getting credit errors, try using a Gemini model instead:")
            print("   python run_original_judge.py gpt5-monitoring-oneoff-100 --model gemini/gemini-2.5-flash")
        elif "gemini" in args.model.lower():
            print("\n💡 Tip: Make sure GOOGLE_API_KEY is set in your environment.")
        return
    
    # Run binary evaluation
    binary_results = None
    if not args.skip_binary:
        print(f"\n{'='*70}")
        print("BINARY EVALUATION (cheating vs legit)")
        print(f"{'='*70}")
        print(f"Evaluating {len(dumps)} samples...")
        
        binary_results = await judge.batch_evaluate_binary(dumps, system_prompt_override=system_prompt_override)
        
        # Save binary results
        binary_output = []
        for i, (result, metadata) in enumerate(zip(binary_results, sample_metadata)):
            binary_output.append({
                'sample_index': i,
                'task_name': metadata['task_name'],
                'sample_id': metadata['sample_id'],
                'sample_original_id': metadata['sample_original_id'],
                'passed': metadata['passed'],
                'score': metadata['score'],
                'judgment': result['judgment'],
                'success': result['success'],
                'raw_response': result['raw_response'] if result['success'] else None,
                'error': result['raw_response'] if not result['success'] else None
            })
        
        binary_file = output_dir / "binary_results.json"
        with open(binary_file, 'w') as f:
            json.dump(binary_output, f, indent=2)
        print(f"\nBinary results saved to: {binary_file}")
        
        # Print stats
        binary_stats = judge.get_stats(binary_results)
        print(f"\nBinary Evaluation Statistics:")
        print(f"  Total samples: {binary_stats['total_samples']}")
        print(f"  Successful evaluations: {binary_stats['successful_evaluations']}")
        print(f"  Success rate: {binary_stats['success_rate']:.1f}%")
        print(f"  Judgment distribution:")
        for judgment, count in binary_stats['judgment_counts'].items():
            print(f"    {judgment}: {count}")
        
        # Check for errors and show sample error message
        if binary_stats['error_count'] > 0:
            print(f"\n⚠️  Warning: {binary_stats['error_count']} evaluations failed!")
            # Show first error as example
            for result in binary_results:
                if not result['success']:
                    error_msg = result['raw_response'][:200]  # First 200 chars
                    print(f"\n  Sample error example: {error_msg}...")
                    if "credit" in error_msg.lower() or "balance" in error_msg.lower():
                        print(f"\n  💡 This looks like an API credit issue. Try using a Gemini model:")
                        print(f"     python run_original_judge.py {args.run_name} --model gemini/gemini-2.5-flash")
                    break
    else:
        print("\nSkipping binary evaluation")
    
    # Run type evaluation on cheating samples
    if not args.skip_type and binary_results:
        cheating_indices = [i for i, r in enumerate(binary_results) if r['judgment'] == 'cheating']
        
        if cheating_indices:
            print(f"\n{'='*70}")
            print("TYPE EVALUATION (cheating classification)")
            print(f"{'='*70}")
            print(f"Evaluating {len(cheating_indices)} samples classified as cheating...")
            
            cheating_dumps = [dumps[i] for i in cheating_indices]
            type_results = await judge.batch_evaluate_type(cheating_dumps, system_prompt_override=system_prompt_override)
            
            # Save type results
            type_output = []
            for i, (result, original_idx) in enumerate(zip(type_results, cheating_indices)):
                metadata = sample_metadata[original_idx]
                binary_result = binary_results[original_idx]
                type_output.append({
                    'sample_index': original_idx,
                    'task_name': metadata['task_name'],
                    'sample_id': metadata['sample_id'],
                    'sample_original_id': metadata['sample_original_id'],
                    'passed': metadata['passed'],
                    'score': metadata['score'],
                    'binary_judgment': binary_result['judgment'],
                    'type_judgment': result['judgment'],
                    'success': result['success'],
                    'raw_response': result['raw_response'] if result['success'] else None,
                    'error': result['raw_response'] if not result['success'] else None
                })
            
            type_file = output_dir / "type_results.json"
            with open(type_file, 'w') as f:
                json.dump(type_output, f, indent=2)
            print(f"\nType results saved to: {type_file}")
            
            # Print stats
            type_stats = judge.get_stats(type_results)
            print(f"\nType Evaluation Statistics:")
            print(f"  Total samples: {type_stats['total_samples']}")
            print(f"  Successful evaluations: {type_stats['successful_evaluations']}")
            print(f"  Success rate: {type_stats['success_rate']:.1f}%")
            print(f"  Cheating type distribution:")
            for judgment, count in type_stats['judgment_counts'].items():
                print(f"    {judgment}: {count}")
        else:
            print("\nNo samples classified as cheating - skipping type evaluation")
    elif args.skip_type:
        print("\nSkipping type evaluation")
    elif not binary_results:
        print("\nCannot run type evaluation without binary results")
    
    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

