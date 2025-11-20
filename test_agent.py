from inspect_ai import eval
from impossiblebench import impossible_swebench



# SWE-bench evaluation with full tool-based scaffold
task = impossible_swebench(
    split="conflicting",   # "original", "oneoff", or "conflicting"
    limit=5,               # Run on first 5 samples (slower)
)

eval(task, model="anthropic/claude-3-5-sonnet-20241022")
