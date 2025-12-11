#!/usr/bin/env python3
"""
Check transcript lengths to find the longest one.

This script scans all eval files and reports transcript token counts.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from inspect_ai.log import read_eval_log
    from inspect_ai.model._chat_message import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool
    from inspect_ai._util.content import ContentText
except ImportError:
    print("Error: inspect_ai not available. Install with: pip install inspect-ai")
    exit(1)


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count.
    Anthropic's tokenizer is closer to ~4-5 chars per token for mixed content.
    We use 4.5 as a conservative estimate.
    """
    return int(len(text) / 4.5)


def get_transcript_from_sample(sample) -> str:
    """Extract full transcript from a sample."""
    transcript = ""
    
    # Get system prompt from task args if available
    if hasattr(sample, 'input') and sample.input:
        transcript += f"**TASK INPUT:**\n{sample.input}\n\n"
    
    # Get all messages (excluding tool outputs for brevity, matching judge_transcripts.py)
    if hasattr(sample, 'messages') and sample.messages:
        for msg in sample.messages:
            # Skip tool outputs - they're too verbose and not needed for judgment
            if isinstance(msg, ChatMessageTool):
                transcript += "**TOOL OUTPUT:** [omitted for brevity]\n\n"
                continue
            
            content = msg.content
            tool_calls = getattr(msg, 'tool_calls', [])
            
            if isinstance(msg, ChatMessageSystem):
                transcript += "**SYSTEM MESSAGE:**\n"
            elif isinstance(msg, ChatMessageUser):
                transcript += "**USER MESSAGE:**\n"
            elif isinstance(msg, ChatMessageAssistant):
                transcript += "**ASSISTANT MESSAGE:**\n"
            
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


def find_eval_files(base_dir: str) -> List[Dict]:
    """Find all eval files to process."""
    base_path = Path(base_dir)
    eval_files = []
    
    # Expected run names
    run_names = [
        "gpt5-default-oneoff-50",
        "gpt5-monitoring-oneoff-50",
        "sonnet37-default-oneoff-50",
        "sonnet37-monitoring-oneoff-50",
        "opus41-default-oneoff-50",
        "opus41-monitoring-oneoff-50",
        "o3-default-oneoff-50",
        "o3-monitoring-oneoff-50"
    ]
    
    for run_name in run_names:
        run_dir = base_path / run_name
        if not run_dir.exists():
            continue
        
        # Find .eval files
        eval_file_paths = list(run_dir.glob("*.eval"))
        for eval_file in eval_file_paths:
            eval_files.append({
                "path": str(eval_file),
                "run_name": run_name,
                "eval_file": eval_file.name
            })
    
    return eval_files


def process_eval_file(eval_file_info: Dict) -> List[Tuple[str, str, int, int]]:
    """Process a single eval file and return transcript lengths."""
    eval_file_path = eval_file_info["path"]
    run_name = eval_file_info["run_name"]
    
    try:
        eval_log = read_eval_log(eval_file_path, header_only=False)
    except Exception as e:
        print(f"Error reading {eval_file_path}: {e}")
        return []
    
    # Extract metadata
    model = eval_log.eval.model
    
    # Get samples
    samples = eval_log.samples if eval_log.samples else []
    
    results = []
    
    for i, sample in enumerate(samples):
        # Use sample.id if available, otherwise try to get from dataset sample_ids
        if hasattr(sample, 'id') and sample.id:
            sample_id = sample.id
        elif hasattr(eval_log, 'eval') and hasattr(eval_log.eval, 'dataset') and eval_log.eval.dataset.sample_ids:
            if i < len(eval_log.eval.dataset.sample_ids):
                sample_id = eval_log.eval.dataset.sample_ids[i]
            else:
                sample_id = f"sample_{i}"
        else:
            sample_id = f"sample_{i}"
        
        # Extract transcript
        transcript = get_transcript_from_sample(sample)
        
        # Calculate length
        char_count = len(transcript)
        token_estimate = estimate_tokens(transcript)
        
        results.append((run_name, sample_id, char_count, token_estimate))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Check transcript lengths")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/impossible_swebench",
        help="Directory containing eval logs"
    )
    
    args = parser.parse_args()
    
    # Find all eval files
    eval_files = find_eval_files(args.logs_dir)
    
    print(f"Found {len(eval_files)} eval files to process")
    print()
    
    all_results = []
    
    # Process each eval file
    for eval_file_info in eval_files:
        results = process_eval_file(eval_file_info)
        all_results.extend(results)
    
    if not all_results:
        print("No transcripts found!")
        return
    
    # Sort by token count
    all_results.sort(key=lambda x: x[3], reverse=True)
    
    print("="*80)
    print("Transcript Length Statistics")
    print("="*80)
    print(f"\nTotal transcripts: {len(all_results)}")
    
    # Calculate statistics
    token_counts = [r[3] for r in all_results]
    char_counts = [r[2] for r in all_results]
    
    print(f"\nToken Count Statistics (estimated):")
    print(f"  Min: {min(token_counts):,} tokens ({min(char_counts):,} chars)")
    print(f"  Max: {max(token_counts):,} tokens ({max(char_counts):,} chars)")
    print(f"  Mean: {sum(token_counts) / len(token_counts):,.0f} tokens ({sum(char_counts) / len(char_counts):,.0f} chars)")
    print(f"  Median: {sorted(token_counts)[len(token_counts)//2]:,} tokens ({sorted(char_counts)[len(char_counts)//2]:,} chars)")
    
    print(f"\nTop 10 Longest Transcripts:")
    print("-" * 80)
    print(f"{'Run Name':<40} {'Sample ID':<30} {'Tokens':<15} {'Chars':<15}")
    print("-" * 80)
    for run_name, sample_id, char_count, token_estimate in all_results[:10]:
        print(f"{run_name:<40} {sample_id:<30} {token_estimate:>14,} {char_count:>14,}")
    
    print(f"\nBottom 10 Shortest Transcripts:")
    print("-" * 80)
    print(f"{'Run Name':<40} {'Sample ID':<30} {'Tokens':<15} {'Chars':<15}")
    print("-" * 80)
    for run_name, sample_id, char_count, token_estimate in all_results[-10:]:
        print(f"{run_name:<40} {sample_id:<30} {token_estimate:>14,} {char_count:>14,}")


if __name__ == "__main__":
    main()

