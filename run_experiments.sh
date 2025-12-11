#!/bin/bash
# Script to run ImpossibleBench experiments across multiple models and prompts
# 
# This script:
# 1. Runs 4 test runs (one per model) with monitoring prompt, 1 task each
# 2. Runs 8 main runs (each model × each prompt) with 50 tasks each on oneoff dataset
#
# Usage:
#   ./run_experiments.sh                    # Run all models (default behavior)
#   ./run_experiments.sh --opus41-only      # Run only opus41 experiments
#   ./run_experiments.sh --monitoring-100   # Skip test runs, only monitoring prompt, 100 tasks per model
#   ./run_experiments.sh --opus41-only --monitoring-100  # Combine flags

set -e  # Exit on error

# Parse arguments
OPUS41_ONLY=false
MONITORING_100=false
for arg in "$@"; do
    if [[ "$arg" == "--opus41-only" ]]; then
        OPUS41_ONLY=true
    elif [[ "$arg" == "--monitoring-100" ]]; then
        MONITORING_100=true
    fi
done

# Define models with their full API names
declare -A MODELS
MODELS["gpt5"]="openai/gpt-5"
MODELS["sonnet37"]="anthropic/claude-3-7-sonnet-20250219"
MODELS["opus41"]="anthropic/claude-opus-4-1-20250805"
MODELS["o3"]="openai/o3"

# Model display names for run names
declare -A MODEL_NAMES
MODEL_NAMES["gpt5"]="gpt5"
MODEL_NAMES["sonnet37"]="sonnet37"
MODEL_NAMES["opus41"]="opus41"
MODEL_NAMES["o3"]="o3"

# Prompts (default behavior: both prompts)
if [[ "$MONITORING_100" == true ]]; then
    PROMPTS=("monitoring")
else
    PROMPTS=("default" "monitoring")
fi

# Determine which models to run
if [[ "$OPUS41_ONLY" == true ]]; then
    MODELS_TO_RUN=("opus41")
else
    MODELS_TO_RUN=("gpt5" "sonnet37" "opus41" "o3")
fi

echo "=========================================="
echo "ImpossibleBench Experiment Runner"
echo "=========================================="
echo ""

# Determine task limit and run name suffix
if [[ "$MONITORING_100" == true ]]; then
    TASK_LIMIT=100
    RUN_NAME_SUFFIX="100"
else
    TASK_LIMIT=50
    RUN_NAME_SUFFIX="50"
fi

# Step 1: Test runs (1 task per model, monitoring prompt) - SKIPPED if --monitoring-100
if [[ "$MONITORING_100" != true ]]; then
    echo "Step 1: Running test runs (1 task per model, monitoring prompt)"
    echo "------------------------------------------------------------"
    for model_key in "${MODELS_TO_RUN[@]}"; do
        model_api="${MODELS[$model_key]}"
        model_name="${MODEL_NAMES[$model_key]}"
        run_name="test-${model_name}-monitoring"
        
        echo ""
        echo "Test run: ${model_name} with monitoring prompt"
        echo "  Model: ${model_api}"
        echo "  Run name: ${run_name}"
        
        python run_agent.py replicate_swe \
            --run-name "${run_name}" \
            --model "${model_api}" \
            --split oneoff \
            --limit 1 \
            --prompt monitoring
        
        echo "  ✓ Completed test run for ${model_name}"
    done
    
    echo ""
    echo "=========================================="
    echo "Test runs completed!"
    echo "=========================================="
    echo ""
fi

# Step 2: Main runs
if [[ "$MONITORING_100" == true ]]; then
    echo "Running main runs (${TASK_LIMIT} tasks each, monitoring prompt only)"
else
    echo "Step 2: Running main runs (${TASK_LIMIT} tasks each, all model×prompt combinations)"
fi
echo "----------------------------------------------------------------------"

for model_key in "${MODELS_TO_RUN[@]}"; do
    model_api="${MODELS[$model_key]}"
    model_name="${MODEL_NAMES[$model_key]}"
    
    for prompt in "${PROMPTS[@]}"; do
        run_name="${model_name}-${prompt}-oneoff-${RUN_NAME_SUFFIX}"
        
        echo ""
        echo "Main run: ${model_name} with ${prompt} prompt"
        echo "  Model: ${model_api}"
        echo "  Prompt: ${prompt}"
        echo "  Tasks: ${TASK_LIMIT}"
        echo "  Run name: ${run_name}"
        
        python run_agent.py replicate_swe \
            --run-name "${run_name}" \
            --model "${model_api}" \
            --split oneoff \
            --limit "${TASK_LIMIT}" \
            --prompt "${prompt}"
        
        echo "  ✓ Completed main run for ${model_name} with ${prompt} prompt"
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
# Calculate summary
NUM_MODELS=${#MODELS_TO_RUN[@]}
NUM_PROMPTS=${#PROMPTS[@]}
if [[ "$MONITORING_100" == true ]]; then
    NUM_TEST_RUNS=0
    NUM_MAIN_RUNS=$((NUM_MODELS * NUM_PROMPTS))
else
    NUM_TEST_RUNS=$NUM_MODELS
    NUM_MAIN_RUNS=$((NUM_MODELS * NUM_PROMPTS))
fi
TOTAL_RUNS=$((NUM_TEST_RUNS + NUM_MAIN_RUNS))

echo "Summary:"
if [[ "$MONITORING_100" != true ]]; then
    echo "  - ${NUM_TEST_RUNS} test run(s) completed"
fi
echo "  - ${NUM_MAIN_RUNS} main run(s) completed (${TASK_LIMIT} tasks each)"
echo "  - Total: ${TOTAL_RUNS} runs"
echo ""
echo "Logs are saved in: logs/impossible_swebench/"

