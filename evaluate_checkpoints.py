#!/usr/bin/env python3
"""Evaluate all models from saved checkpoints with optimized inference."""

import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate models from checkpoints')
    parser.add_argument('--config', type=str, default='configs/quick_test.yml',
                        help='Configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_test',
                        help='Directory containing checkpoints')
    parser.add_argument('--fast-inference', action='store_true',
                        help='Use fast inference mode for AME-ODE')
    
    args = parser.parse_args()
    
    # Run the comparison script with checkpoint evaluation
    cmd = [
        sys.executable, 'compare_baselines.py',
        '--config', args.config,
        '--system', args.system,
        '--use-checkpoints',
        '--checkpoint-dir', args.checkpoint_dir
    ]
    
    if args.fast_inference:
        cmd.append('--fast-inference')
    
    print("Evaluating all models from checkpoints...")
    if args.fast_inference:
        print("Using fast inference mode for AME-ODE")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()