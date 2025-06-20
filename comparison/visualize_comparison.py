"""
Visualize comparison results between Wave-U-Net and other models.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_comparison(results_file, output_dir=None):
    """
    Create visualizations of model comparison results.
    
    Args:
        results_file: Path to JSON file with comparison results
        output_dir: Directory to save visualizations (defaults to same directory as results)
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    models = list(results.keys())
    metrics = ["SDR", "SI-SDR", "SNR"]
    
    # 1. Average Performance Comparison
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [results[model]["average"][metric] for metric in metrics]
        plt.bar(x + i*width - width/2, values, width, label=model.capitalize())
    
    plt.xlabel('Metrics')
    plt.ylabel('dB')
    plt.title('Model Comparison: Average Performance')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            value = results[model]["average"][metric]
            plt.text(j + i*width - width/2, value + 0.1, f'{value:.2f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_average.png'), dpi=300)
    plt.close()
    
    # 2. Per-track SDR Comparison
    plt.figure(figsize=(12, 6))
    
    track_names = [track["track_name"] for track in results["waveunet"]["tracks"]]
    num_tracks = len(track_names)
    
    x = np.arange(num_tracks)
    width = 0.35
    
    for i, model in enumerate(models):
        values = [track["metrics"]["SDR"] for track in results[model]["tracks"]]
        plt.bar(x + i*width - width/2, values, width, label=model.capitalize())
    
    plt.xlabel('Tracks')
    plt.ylabel('SDR (dB)')
    plt.title('Model Comparison: Per-track SDR Performance')
    plt.xticks(x, track_names, rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_sdr.png'), dpi=300)
    plt.close()
    
    # 3. Metric Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        for model in models:
            values = [track["metrics"][metric] for track in results[model]["tracks"] 
                     if track["metrics"][metric] is not None and 
                     not np.isinf(track["metrics"][metric]) and 
                     not np.isnan(track["metrics"][metric])]
            
            axes[i].hist(values, bins=10, alpha=0.7, label=model.capitalize())
        
        axes[i].set_title(f'{metric} Distribution')
        axes[i].set_xlabel(f'{metric} (dB)')
        axes[i].set_ylabel('Count')
        axes[i].grid(linestyle='--', alpha=0.7)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model comparison results")
    parser.add_argument('--results_file', type=str, default='comparison_results/comparison_results.json',
                        help='Path to comparison results JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualize_comparison(args.results_file, args.output_dir)


if __name__ == "__main__":
    main()