#!/bin/bash
# Run all ablation studies for AME-ODE

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Make sure we're in the right directory
cd "$(dirname "$0")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           AME-ODE Ablation Studies Runner                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if configs exist
if [ ! -d "configs/ablation" ]; then
    echo -e "${RED}Error: configs/ablation directory not found!${NC}"
    exit 1
fi

# Parse command line arguments
NUM_GPUS=${1:-1}
CATEGORY=${2:-all}
SYSTEM=${3:-multi_scale_oscillators}

# Show all available configs
echo -e "${YELLOW}Available ablation configurations:${NC}"
echo "─────────────────────────────────────"

# Group configs by category
echo -e "\n${GREEN}Routing Mechanism:${NC}"
ls -1 configs/ablation/routing_*.yml 2>/dev/null | while read f; do
    basename "$f" | sed 's/^/  • /'
done

echo -e "\n${GREEN}Number of Experts:${NC}"
ls -1 configs/ablation/experts_*.yml 2>/dev/null | while read f; do
    basename "$f" | sed 's/^/  • /'
done

echo -e "\n${GREEN}Temperature:${NC}"
ls -1 configs/ablation/temp_*.yml 2>/dev/null | while read f; do
    basename "$f" | sed 's/^/  • /'
done

echo -e "\n${GREEN}Regularization:${NC}"
ls -1 configs/ablation/reg_*.yml 2>/dev/null | while read f; do
    basename "$f" | sed 's/^/  • /'
done

echo -e "\n${GREEN}Base Configuration:${NC}"
ls -1 configs/ablation/base_*.yml 2>/dev/null | while read f; do
    basename "$f" | sed 's/^/  • /'
done

# Count total configs
num_configs=$(ls configs/ablation/*.yml 2>/dev/null | wc -l)
echo -e "\n${BLUE}Total configurations: $num_configs${NC}"
echo "─────────────────────────────────────"

# Show run parameters
echo -e "\n${YELLOW}Run Parameters:${NC}"
echo "  • GPUs: $NUM_GPUS"
echo "  • Category: $CATEGORY"
echo "  • System: $SYSTEM"
echo ""

# Confirm before running
read -p "Continue with ablation studies? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting ablation studies...${NC}"
echo "════════════════════════════════════════"

# Create output directory
mkdir -p ablation_results

# Log file
LOG_FILE="ablation_results/ablation_run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Function to run with progress
run_with_progress() {
    local cmd="$1"
    local desc="$2"
    
    echo -e "${YELLOW}Running: $desc${NC}"
    echo "Command: $cmd"
    echo ""
    
    # Run command and capture output
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    
    return ${PIPESTATUS[0]}
}

# Run the ablations
CMD="python experiments/run_config_ablations.py \
    --category $CATEGORY \
    --config-dir configs/ablation \
    --output-dir ablation_results \
    --num-gpus $NUM_GPUS"

run_with_progress "$CMD" "Ablation experiments for category: $CATEGORY"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Ablation studies completed successfully!${NC}"
    echo "Results saved in: ablation_results/"
    
    # Show summary of results
    echo ""
    echo -e "${YELLOW}Results Summary:${NC}"
    echo "─────────────────────────────────────"
    
    # Find and display result files
    if [ -d "ablation_results/$CATEGORY" ]; then
        echo -e "\n${GREEN}Generated files:${NC}"
        find "ablation_results/$CATEGORY" -name "*.json" -type f | while read f; do
            echo "  • $(basename "$f")"
        done
    fi
    
    # Show comparison if all categories were run
    if [ "$CATEGORY" = "all" ]; then
        echo ""
        echo -e "${YELLOW}Running cross-category comparison...${NC}"
        python experiments/run_config_ablations.py --compare --output-dir ablation_results
    fi
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    Complete! 🎉                            ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}✗ Error: Ablation studies failed!${NC}"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi

# Show how to visualize results
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. View results:        cat ablation_results/$CATEGORY/summary.json | jq"
echo "  2. Compare categories:  python experiments/run_config_ablations.py --compare"
echo "  3. Generate plots:      python experiments/plot_ablation_results.py"
echo ""