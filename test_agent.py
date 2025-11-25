from inspect_ai import eval
from impossiblebench import impossible_swebench



# SWE-bench evaluation with full tool-based scaffold
task = impossible_swebench(
    split="conflicting",   # "original", "oneoff", or "conflicting"
    # limit=5,               # Run on first 5 samples (slower)
)

eval(task, model="anthropic/claude-haiku-4-5-20251001")
