"""
ImpossibleBench: Testing AI systems on impossible and edge-case scenarios.

This module contains solvers and datasets for testing AI systems on:
- SWE-bench impossible patches (conflicting, inverted, one-off)
- HumanEval/LiveCodeBench with impossible test cases
"""

# SWE-bench task imports
from .swebench_tasks import impossible_swebench
from .swebench_scorers import swe_bench_scorer

# SWE-bench agent imports
from .swebench_agent_full import multi_submission_solver



__all__ = [
    # SWE-bench tasks
    "impossible_swebench",
    "swe_bench_scorer",

    # SWE-bench agents
    "multi_submission_solver",
]
