#!/bin/bash
# Run all ablation studies for AME-ODE

# Make sure we're in the right directory
cd "$(dirname "$0")"

echo "Running AME-ODE Ablation Studies"
echo "================================"
echo ""

# Check if configs exist
if [ ! -d "configs/ablation" ]; then
    echo "Error: configs/ablation directory not found!"
    exit 1
fi

# Count config files
num_configs=$(ls configs/ablation/*.yml 2>/dev/null | wc -l)
echo "Found $num_configs ablation configurations"
echo ""

# Parse command line arguments
NUM_GPUS=${1:-1}
CATEGORY=${2:-all}

echo "Using $NUM_GPUS GPU(s)"
echo "Running category: $CATEGORY"
echo ""

# Run the ablations
python experiments/run_config_ablations.py \
    --category $CATEGORY \
    --config-dir configs/ablation \
    --output-dir ablation_results \
    --num-gpus $NUM_GPUS

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Ablation studies completed successfully!"
    echo "Results saved in: ablation_results/"
    
    # Show comparison if all categories were run
    if [ "$CATEGORY" = "all" ]; then
        echo ""
        echo "Running cross-category comparison..."
        python experiments/run_config_ablations.py --compare --output-dir ablation_results
    fi
else
    echo ""
    echo "Error: Ablation studies failed!"
    exit 1
fi