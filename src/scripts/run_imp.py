import argparse
from inspect_ai import eval
from impossiblebench import impossible_swebench

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tasks to run (default: all tasks)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt type (e.g., 'predict_eval' to use eval prompt version)",
    )
    args = parser.parse_args()

    task = impossible_swebench(
        split="conflicting",
        agent_type="tools",
        limit=args.limit,
        prompt_type=args.prompt,
    )

    eval(task, model="anthropic/claude-3-7-sonnet-20250219", log_dir="./logs")


if __name__ == "__main__":
    main()
