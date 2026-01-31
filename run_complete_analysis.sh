#!/bin/bash
# Complete analysis workflow
cd /home/admin1/Desktop/CRC-Select-Torch
echo "Complete Multi-Seed Analysis"
echo "============================="
./manual_checkpoint_setup.sh && \
./run_eval_all_seeds.sh && \
python3 scripts/compute_violation_rate.py --method_dirs ../results_paper/CRC-Select --seeds 42 123 456 789 --alphas 0.1 --generate_latex && \
python3 scripts/compare_ood_safety.py --methods CRC-Select --seeds 42 123 456 789 --plot --latex
echo "Done!"
