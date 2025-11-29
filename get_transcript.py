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
    print(df.columns)
    df = df[df['file_path'] == 'logs/2025-11-20T19-27-51-05-00_swebench-conflicting_FffjzuvM9S6rZPWwYT3Uyr.eval']
    print(df.iloc[0]['dump'])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    # Force exit - httpx connections may not close properly
    sys.exit(0)
