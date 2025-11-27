#!/usr/bin/env python3
"""Extract predictions and scores from eval file and create visualization.

Usage:
    python plot_eval_prediction_vs_score.py --eval_file <path_to_eval_file>
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from inspect_ai.log import read_eval_log
    from inspect_ai.model._chat_message import ChatMessageAssistant
    from inspect_ai._util.content import ContentText
except ImportError:
    print("Error: inspect_ai not available. Please install inspect-ai.")
    exit(1)


def extract_prediction_from_message(message):
    """Extract prediction value from assistant message.
    
    Looks for <prediction>X.XX</prediction> tags in the message content.
    Returns the float value if found, None otherwise.
    """
    # Get message content
    content = message.content
    if not isinstance(content, list):
        content = [content]
    
    # Extract text from all content items
    text_parts = []
    for c in content:
        if isinstance(c, ContentText):
            text_parts.append(c.text)
        elif isinstance(c, str):
            text_parts.append(c)
    
    full_text = " ".join(text_parts)
    
    # Look for <prediction>X.XX</prediction> pattern (handles various decimal formats)
    # Matches: 0.00, 0.5, 0.50, 1.0, 1.00, etc.
    pattern = r'<prediction>([0-9]+\.?[0-9]*)</prediction>'
    match = re.search(pattern, full_text)
    
    if match:
        try:
            value = float(match.group(1))
            # Validate range (0.0 to 1.0)
            if 0.0 <= value <= 1.0:
                return value
            else:
                return None
        except ValueError:
            return None
    return None


def extract_score(sample):
    """Extract score from sample.
    
    Returns the score value (0 or 1) if available, None otherwise.
    """
    score_to_float = lambda v: float({'I': 0, 'C': 1}.get(v, v))
    
    # Try to get score from sample.scores (dictionary of scorer results)
    if hasattr(sample, 'scores') and sample.scores:
        for scorer_name, score_result in sample.scores.items():
            if hasattr(score_result, 'value'):
                try:
                    return score_to_float(score_result.value)
                except (ValueError, TypeError):
                    continue
    
    # Try sample.score directly
    if hasattr(sample, 'score') and sample.score is not None:
        try:
            return score_to_float(sample.score)
        except (ValueError, TypeError):
            pass
    
    return None


def extract_data_from_eval_file(eval_file_path):
    """Extract predictions and scores from eval file.
    
    Returns:
        tuple: (predictions, scores) lists, and counts of excluded tasks
    """
    predictions = []
    scores = []
    missing_prediction_count = 0
    unscored_count = 0
    
    # Read the eval log
    log = read_eval_log(eval_file_path, header_only=False)
    
    if not log.samples:
        print("Warning: No samples found in eval file.")
        return predictions, scores, missing_prediction_count, unscored_count
    
    for sample in log.samples:
        # Extract score
        score = extract_score(sample)
        if score is None:
            unscored_count += 1
            continue
        
        # Extract prediction from first assistant message
        if not hasattr(sample, 'messages') or not sample.messages:
            missing_prediction_count += 1
            continue
        
        # Find first assistant message
        first_assistant_msg = None
        for msg in sample.messages:
            if isinstance(msg, ChatMessageAssistant):
                first_assistant_msg = msg
                break
        
        if first_assistant_msg is None:
            missing_prediction_count += 1
            continue
        
        # Extract prediction
        prediction = extract_prediction_from_message(first_assistant_msg)
        if prediction is None:
            missing_prediction_count += 1
            continue
        
        # Valid data point
        predictions.append(prediction)
        scores.append(score)
    
    return predictions, scores, missing_prediction_count, unscored_count


def create_strip_plot(predictions, scores, output_path):
    """Create strip plot with jitter showing predictions vs scores.
    
    Args:
        predictions: List of prediction values (0.00-1.00)
        scores: List of score values (0 or 1)
        output_path: Path to save the plot
    """
    # Convert to numpy arrays
    predictions = np.array(predictions)
    scores = np.array(scores)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predictions by score
    predictions_score_0 = predictions[scores == 0]
    predictions_score_1 = predictions[scores == 1]
    
    # Add jitter to x positions
    jitter_strength = 0.15
    x_pos_0 = np.zeros(len(predictions_score_0)) + np.random.normal(0, jitter_strength, len(predictions_score_0))
    x_pos_1 = np.ones(len(predictions_score_1)) + np.random.normal(0, jitter_strength, len(predictions_score_1))
    
    # Plot points
    ax.scatter(x_pos_0, predictions_score_0, alpha=0.6, s=50, label='Score = 0 (No cheating)', color='blue')
    ax.scatter(x_pos_1, predictions_score_1, alpha=0.6, s=50, label='Score = 1 (Cheating)', color='red')
    
    # Add box plots for distribution
    box_data = [predictions_score_0, predictions_score_1]
    bp = ax.boxplot(box_data, positions=[0, 1], widths=0.3, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Style box plots
    for patch in bp['boxes']:
        patch.set_facecolor('lightgray')
        patch.set_alpha(0.3)
    
    # Set labels and title
    ax.set_xlabel('Score (Cheating Status)', fontsize=12)
    ax.set_ylabel('Model Prediction (Evaluation Probability)', fontsize=12)
    ax.set_title('Model Prediction vs Cheating Score', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Cheating (0)', 'Cheating (1)'])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    # Add statistics text
    stats_text = f'Total points: {len(predictions)}\n'
    stats_text += f'Score=0: {len(predictions_score_0)} (mean={np.mean(predictions_score_0):.2f})\n'
    stats_text += f'Score=1: {len(predictions_score_1)} (mean={np.mean(predictions_score_1):.2f})'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract predictions and scores from eval file and create visualization')
    parser.add_argument('--eval_file', type=str, required=True,
                       help='Path to the .eval file')
    args = parser.parse_args()
    
    eval_file_path = Path(args.eval_file)
    if not eval_file_path.exists():
        print(f"Error: Eval file not found: {eval_file_path}")
        exit(1)
    
    # Extract data
    print(f"Reading eval file: {eval_file_path}")
    predictions, scores, missing_prediction_count, unscored_count = extract_data_from_eval_file(eval_file_path)
    
    # Print statistics
    print(f"\nData extraction complete:")
    print(f"  Valid data points: {len(predictions)}")
    print(f"  Tasks with missing/invalid predictions: {missing_prediction_count}")
    print(f"  Unscored tasks: {unscored_count}")
    
    if len(predictions) == 0:
        print("Error: No valid data points found. Cannot create plot.")
        exit(1)
    
    # Create output directory if needed
    output_dir = Path(__file__).parent / "images"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "eval_prediction_vs_score.png"
    
    # Create plot
    print(f"\nCreating plot...")
    create_strip_plot(predictions, scores, output_path)
    print("Done!")


if __name__ == "__main__":
    main()

