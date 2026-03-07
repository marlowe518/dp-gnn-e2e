"""Analyze and visualize epsilon impact on DP-MLP pretraining.

Usage:
    python scripts/analyze_epsilon_impact.py --results-dir results/experiments/epsilon_sweep_transductive
"""

import argparse
import json
import glob
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> List[Dict]:
    """Load all result JSON files."""
    result_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    results = []
    for f in result_files:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    return results


def create_epsilon_vs_accuracy_plot(results: List[Dict], output_path: str):
    """Create epsilon vs accuracy plot."""
    # Separate DP and non-DP results
    dp_results = [r for r in results if r['final_epsilon'] != float('inf')]
    non_dp_results = [r for r in results if r['final_epsilon'] == float('inf')]
    
    if not dp_results:
        print("No DP results found")
        return
    
    # Sort by epsilon
    dp_results.sort(key=lambda x: x['final_epsilon'])
    
    # Extract data
    epsilons = [r['final_epsilon'] for r in dp_results]
    test_accs = [r['final_test_acc'] * 100 for r in dp_results]
    val_accs = [r['final_val_acc'] * 100 for r in dp_results]
    train_accs = [r['final_train_acc'] * 100 for r in dp_results]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(epsilons, test_accs, 'o-', label='Test Accuracy', linewidth=2, markersize=8)
    ax.semilogx(epsilons, val_accs, 's-', label='Val Accuracy', linewidth=2, markersize=8)
    ax.semilogx(epsilons, train_accs, '^-', label='Train Accuracy', linewidth=2, markersize=8)
    
    # Add non-DP baseline
    if non_dp_results:
        non_dp_acc = non_dp_results[0]['final_test_acc'] * 100
        ax.axhline(y=non_dp_acc, color='r', linestyle='--', label=f'Non-DP Test ({non_dp_acc:.1f}%)')
    
    ax.set_xlabel('Epsilon (Privacy Budget)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Privacy-Utility Trade-off: DP-MLP Pretraining', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")


def create_gap_analysis_plot(results: List[Dict], output_path: str):
    """Create gap analysis plot."""
    # Get non-DP baseline
    non_dp_results = [r for r in results if r['final_epsilon'] == float('inf')]
    if not non_dp_results:
        print("No non-DP baseline found")
        return
    
    non_dp_acc = non_dp_results[0]['final_test_acc']
    
    # Filter DP results
    dp_results = [r for r in results if r['final_epsilon'] != float('inf')]
    dp_results.sort(key=lambda x: x['final_epsilon'])
    
    if not dp_results:
        print("No DP results found")
        return
    
    # Calculate gaps
    epsilons = [r['final_epsilon'] for r in dp_results]
    gaps = [(non_dp_acc - r['final_test_acc']) * 100 for r in dp_results]
    pct_of_non_dp = [(r['final_test_acc'] / non_dp_acc) * 100 for r in dp_results]
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.semilogx(epsilons, gaps, 'o-', color=color1, linewidth=2, markersize=8)
    ax1.set_xlabel('Epsilon (Privacy Budget)', fontsize=12)
    ax1.set_ylabel('Accuracy Gap to Non-DP (%)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.semilogx(epsilons, pct_of_non_dp, 's--', color=color2, linewidth=2, markersize=8)
    ax2.set_ylabel('% of Non-DP Performance', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)
    
    plt.title('DP Performance Gap Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")


def create_steps_vs_epsilon_plot(results: List[Dict], output_path: str):
    """Create steps vs epsilon plot."""
    dp_results = [r for r in results if r['final_epsilon'] != float('inf')]
    dp_results.sort(key=lambda x: x['final_epsilon'])
    
    if not dp_results:
        print("No DP results found")
        return
    
    epsilons = [r['final_epsilon'] for r in dp_results]
    steps = [r['steps_taken'] for r in dp_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(epsilons, steps, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Epsilon (Privacy Budget)', fontsize=12)
    ax.set_ylabel('Training Steps Completed', fontsize=12)
    ax.set_title('Training Steps vs Privacy Budget', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")


def generate_report(results: List[Dict], output_path: str):
    """Generate markdown report."""
    # Get non-DP baseline
    non_dp_results = [r for r in results if r['final_epsilon'] == float('inf')]
    non_dp_acc = non_dp_results[0]['final_test_acc'] * 100 if non_dp_results else 0
    
    # Sort DP results
    dp_results = [r for r in results if r['final_epsilon'] != float('inf')]
    dp_results.sort(key=lambda x: x['final_epsilon'])
    
    report = []
    report.append("# Epsilon Impact Analysis Report\n")
    report.append("## DP-MLP Pretraining on ogbn-arxiv\n")
    report.append(f"**Date:** {np.datetime64('now')}\n")
    report.append(f"**Non-DP Baseline Test Accuracy:** {non_dp_acc:.2f}%\n\n")
    
    report.append("## Results Summary\n\n")
    report.append("| Epsilon | Test Acc | Val Acc | Train Acc | Steps | Gap to Non-DP | % of Non-DP |\n")
    report.append("|---------|----------|---------|-----------|-------|---------------|-------------|\n")
    
    for r in dp_results:
        eps = r['final_epsilon']
        test_acc = r['final_test_acc'] * 100
        val_acc = r['final_val_acc'] * 100
        train_acc = r['final_train_acc'] * 100
        steps = r['steps_taken']
        gap = non_dp_acc - test_acc
        pct = (test_acc / non_dp_acc) * 100 if non_dp_acc > 0 else 0
        report.append(f"| {eps:.2f} | {test_acc:.2f}% | {val_acc:.2f}% | {train_acc:.2f}% | {steps} | {gap:.2f}% | {pct:.1f}% |\n")
    
    report.append(f"| Non-DP | {non_dp_acc:.2f}% | - | - | - | 0.00% | 100.0% |\n\n")
    
    report.append("## Key Findings\n\n")
    
    if dp_results:
        # Find best DP result
        best_dp = max(dp_results, key=lambda x: x['final_test_acc'])
        report.append(f"1. **Best DP Performance:** ε={best_dp['final_epsilon']:.2f} achieves {best_dp['final_test_acc']*100:.2f}% test accuracy\n")
        
        # Calculate improvement from lowest to highest epsilon
        lowest = min(dp_results, key=lambda x: x['final_epsilon'])
        highest = max(dp_results, key=lambda x: x['final_epsilon'])
        improvement = (highest['final_test_acc'] - lowest['final_test_acc']) * 100
        report.append(f"2. **Epsilon Impact:** Increasing ε from {lowest['final_epsilon']:.2f} to {highest['final_epsilon']:.2f} "
                     f"improves accuracy by {improvement:.2f}%\n")
        
        # Check if trend is as expected
        if highest['final_test_acc'] > lowest['final_test_acc']:
            report.append(f"3. **Trend Validation:** ✓ Higher epsilon leads to better performance (as expected)\n")
        else:
            report.append(f"3. **Trend Validation:** ✗ No clear trend between epsilon and performance\n")
        
        report.append(f"4. **Gap to Non-DP:** Even at highest epsilon ({highest['final_epsilon']:.2f}), "
                     f"DP performance is {(non_dp_acc - highest['final_test_acc']*100):.2f}% below non-DP baseline\n")
    
    report.append("\n## Hyperparameters Used\n\n")
    if dp_results:
        sample = dp_results[0]
        report.append(f"- **Noise Multiplier:** {sample['config'].get('noise_multiplier', 'N/A')}\n")
        report.append(f"- **Batch Size:** {sample['config'].get('batch_size', 'N/A')}\n")
        report.append(f"- **Learning Rate:** {sample['config'].get('learning_rate', 'N/A')}\n")
        report.append(f"- **Epochs:** {sample['config'].get('num_epochs', 'N/A')}\n")
        report.append(f"- **Latent Size:** {sample['config'].get('latent_size', 'N/A')}\n")
        report.append(f"- **Num Layers:** {sample['config'].get('num_layers', 'N/A')}\n")
    
    report.append("\n## Recommendations\n\n")
    report.append("1. **Hyperparameter Tuning:** Consider increasing batch size and reducing noise multiplier for better utility-privacy trade-off\n")
    report.append("2. **Training Duration:** DP training may require more epochs to converge\n")
    report.append("3. **Alternative Approaches:** Consider DP-Adam or other DP optimizers\n")
    report.append("4. **Lower Epsilon Ranges:** The tested epsilon values may be too high; consider exploring ε < 1\n")
    
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze epsilon impact')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots and report')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")
    
    # Create visualizations
    create_epsilon_vs_accuracy_plot(
        results, 
        os.path.join(args.output_dir, 'epsilon_vs_accuracy.png')
    )
    
    create_gap_analysis_plot(
        results,
        os.path.join(args.output_dir, 'gap_analysis.png')
    )
    
    create_steps_vs_epsilon_plot(
        results,
        os.path.join(args.output_dir, 'steps_vs_epsilon.png')
    )
    
    # Generate report
    generate_report(
        results,
        os.path.join(args.output_dir, 'analysis_report.md')
    )
    
    print(f"\nAnalysis complete! Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
