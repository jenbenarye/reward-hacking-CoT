#!/usr/bin/env python3
"""
Read a specific transcript from an eval file.
"""

import argparse
import sys
from pathlib import Path

try:
    from inspect_ai.log import read_eval_log
    from inspect_ai.model._chat_message import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool
    from inspect_ai._util.content import ContentText
except ImportError:
    print("Error: inspect_ai not available. Install with: pip install inspect-ai")
    exit(1)


def get_transcript_from_sample(sample) -> str:
    """Extract full transcript from a sample."""
    transcript = ""
    
    # Get system prompt from task args if available
    if hasattr(sample, 'input') and sample.input:
        transcript += f"**TASK INPUT:**\n{sample.input}\n\n"
    
    # Get all messages
    if hasattr(sample, 'messages') and sample.messages:
        for msg in sample.messages:
            content = msg.content
            tool_calls = getattr(msg, 'tool_calls', [])
            
            if isinstance(msg, ChatMessageSystem):
                transcript += "**SYSTEM MESSAGE:**\n"
            elif isinstance(msg, ChatMessageUser):
                transcript += "**USER MESSAGE:**\n"
            elif isinstance(msg, ChatMessageAssistant):
                transcript += "**ASSISTANT MESSAGE:**\n"
            elif isinstance(msg, ChatMessageTool):
                transcript += "**TOOL OUTPUT:**\n"
            
            if not isinstance(content, list):
                content = [content]
            for c in content:
                if isinstance(c, ContentText):
                    transcript += c.text + "\n"
                elif isinstance(c, str):
                    transcript += c + "\n"
            
            if tool_calls:
                for tool_call in tool_calls:
                    transcript += f"[TOOL CALL: {tool_call.function}, ARGS: {tool_call.arguments}]\n"
            
            transcript += "\n"
    
    # Get test results from events if available
    # Only include score events - other events (sandbox, logger, span, etc.) are too verbose
    if hasattr(sample, 'events') and sample.events:
        score_events = [event for event in sample.events if hasattr(event, 'score')]
        if score_events:
            transcript += "**TEST SCORE EVENTS:**\n"
            for event in score_events:
                score_value = event.score.value if hasattr(event.score, 'value') else str(event.score)
                score_explanation = event.score.explanation if hasattr(event.score, 'explanation') else ""
                transcript += f"Score: {score_value}"
                if score_explanation:
                    transcript += f" - {score_explanation}"
                transcript += "\n"
            transcript += "\n"
    
    # Get scores if available
    if hasattr(sample, 'scores') and sample.scores:
        transcript += "**FINAL SCORES:**\n"
        for scorer_name, score in sample.scores.items():
            score_value = score.value if hasattr(score, 'value') else str(score)
            score_explanation = score.explanation if hasattr(score, 'explanation') else ""
            transcript += f"{scorer_name}: {score_value}"
            if score_explanation:
                transcript += f" - {score_explanation}"
            transcript += "\n"
        transcript += "\n"
    
    # Get metadata about test files if available
    if hasattr(sample, 'metadata') and sample.metadata:
        if 'test_patch_files' in sample.metadata:
            transcript += f"**TEST FILES MODIFIED BY TEST PATCH:**\n{sample.metadata['test_patch_files']}\n\n"
    
    return transcript


def find_sample_in_eval_file(eval_file_path: str, sample_id: str):
    """Find a specific sample in an eval file."""
    try:
        eval_log = read_eval_log(eval_file_path, header_only=False)
    except Exception as e:
        print(f"Error reading {eval_file_path}: {e}")
        return None
    
    samples = eval_log.samples if eval_log.samples else []
    
    # Try to find by sample.id first
    for sample in samples:
        if hasattr(sample, 'id') and sample.id == sample_id:
            return sample, eval_log
    
    # Try to find by dataset sample_ids
    if hasattr(eval_log, 'eval') and hasattr(eval_log.eval, 'dataset') and eval_log.eval.dataset.sample_ids:
        for i, dataset_sample_id in enumerate(eval_log.eval.dataset.sample_ids):
            if dataset_sample_id == sample_id and i < len(samples):
                return samples[i], eval_log
    
    return None, eval_log


def main():
    parser = argparse.ArgumentParser(description="Read a specific transcript from an eval file")
    parser.add_argument(
        "run_name",
        help="Run name (e.g., sonnet37-default-oneoff-50)"
    )
    parser.add_argument(
        "sample_id",
        help="Sample ID (e.g., pydata__xarray-4075)"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/impossible_swebench",
        help="Directory containing eval logs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save transcript (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Find eval file
    base_path = Path(args.logs_dir)
    run_dir = base_path / args.run_name
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Find .eval files
    eval_file_paths = list(run_dir.glob("*.eval"))
    if not eval_file_paths:
        print(f"Error: No .eval files found in {run_dir}")
        sys.exit(1)
    
    if len(eval_file_paths) > 1:
        print(f"Warning: Multiple .eval files found, using: {eval_file_paths[0]}")
    
    eval_file_path = str(eval_file_paths[0])
    
    print(f"Reading transcript from: {eval_file_path}")
    print(f"Looking for sample: {args.sample_id}")
    print()
    
    # Find the sample
    sample, eval_log = find_sample_in_eval_file(eval_file_path, args.sample_id)
    
    if sample is None:
        print(f"Error: Sample '{args.sample_id}' not found in {eval_file_path}")
        print(f"\nAvailable sample IDs:")
        if hasattr(eval_log, 'eval') and hasattr(eval_log.eval, 'dataset') and eval_log.eval.dataset.sample_ids:
            for sid in eval_log.eval.dataset.sample_ids[:10]:
                print(f"  - {sid}")
            if len(eval_log.eval.dataset.sample_ids) > 10:
                print(f"  ... and {len(eval_log.eval.dataset.sample_ids) - 10} more")
        sys.exit(1)
    
    # Extract transcript
    transcript = get_transcript_from_sample(sample)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"Transcript saved to: {args.output}")
        print(f"Length: {len(transcript):,} characters (~{len(transcript) // 4.5:,.0f} tokens)")
    else:
        print("="*80)
        print("TRANSCRIPT")
        print("="*80)
        print(transcript)
        print("="*80)
        print(f"\nLength: {len(transcript):,} characters (~{len(transcript) // 4.5:,.0f} tokens)")


if __name__ == "__main__":
    main()

