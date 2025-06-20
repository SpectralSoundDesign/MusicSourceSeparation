import json
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_evaluation_results(results_file='evaluation_results/evaluation_results.json'):
    """
    Create visualizations of evaluation results.
    
    Args:
        results_file: Path to JSON file with evaluation results
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    individual_results = results['individual_tracks']
    avg_metrics = results['average_metrics']
    
    # Create output directory for visualizations
    output_dir = os.path.dirname(results_file)
    
    # Plot individual track results
    metrics = ['SDR', 'SI-SDR', 'SNR']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [track['metrics'][metric] for track in individual_results 
                  if track['metrics'][metric] is not None]
        valid_tracks = [j for j, track in enumerate(individual_results) 
                       if track['metrics'][metric] is not None]
        
        if values:
            axes[i].bar(valid_tracks, values)
            axes[i].set_title(f'{metric} by Track')
            axes[i].set_xlabel('Track ID')
            axes[i].set_ylabel(f'{metric} (dB)')
            if not np.isnan(avg_metrics[metric]):
                axes[i].axhline(y=avg_metrics[metric], color='r', linestyle='--', 
                               label=f'Average: {avg_metrics[metric]:.2f} dB')
                axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
            axes[i].set_title(f'{metric} by Track')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_metrics.png'), dpi=300)
    plt.close()
    
    # Plot metric distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [track['metrics'][metric] for track in individual_results 
                  if track['metrics'][metric] is not None and 
                  not np.isinf(track['metrics'][metric])]
        
        if values:
            axes[i].hist(values, bins=10, edgecolor='black')
            axes[i].set_title(f'{metric} Distribution')
            axes[i].set_xlabel(f'{metric} (dB)')
            axes[i].set_ylabel('Count')
            if not np.isnan(avg_metrics[metric]):
                axes[i].axvline(x=avg_metrics[metric], color='r', linestyle='--', 
                               label=f'Average: {avg_metrics[metric]:.2f} dB')
                axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
            axes[i].set_title(f'{metric} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distribution.png'), dpi=300)
    plt.close()
    
    # Create summary plot
    plt.figure(figsize=(8, 6))
    valid_metrics = [(m, avg_metrics[m]) for m in metrics if not np.isnan(avg_metrics[m])]
    
    if valid_metrics:
        metric_names, metric_values = zip(*valid_metrics)
        bars = plt.bar(metric_names, metric_values)
        plt.ylabel('dB')
        plt.title('Average Performance Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.2f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No valid metrics available', ha='center', va='center')
        plt.title('Average Performance Metrics - No Data')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument('--results_file', type=str, 
                        default='evaluation_results/evaluation_results.json',
                        help='Path to evaluation results JSON file')
    
    args = parser.parse_args()
    
    visualize_evaluation_results(args.results_file)