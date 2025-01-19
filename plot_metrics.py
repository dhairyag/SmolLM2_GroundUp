import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_and_plot_metrics(metrics_file="checkpoints/metrics_final.pt"):
    """
    Load and plot training metrics including loss and unique content count.
    
    The unique content count represents the number of distinct training examples 
    seen during training. A steadily increasing count indicates the model is being
    exposed to new, diverse content, while plateaus might suggest data repetition
    or sampling issues.
    """
    # Load metrics
    metrics = torch.load(metrics_file)
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss with log scale on y-axis
    ax1.semilogy(metrics['steps'], metrics['loss'], linewidth=2)
    ax1.set_title('Training Loss vs Steps (Log Scale)', pad=15)
    ax1.set_xlabel('Steps', labelpad=10)
    ax1.set_ylabel('Loss (log scale)', labelpad=10)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.grid(True, which="minor", ls=":", alpha=0.2)
    # Plot 2: Unique content count
    ax2.plot(metrics['steps'], metrics['unique_content_count'], linewidth=2, color='green', marker='o', markersize=4)
    ax2.set_title('Unique Training Examples Over Time', pad=15)
    ax2.set_xlabel('Steps', labelpad=10)
    ax2.set_ylabel('Number of Unique Examples', labelpad=10)
    ax2.grid(True, alpha=0.2)
    # Add some spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    
    # Save plot with high DPI for better quality
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Total steps: {len(metrics['steps'])}")
    print(f"Final loss: {metrics['loss'][-1]:.4f}")
    print(f"Min loss: {min(metrics['loss']):.4f}")
    print(f"Total unique examples seen: {metrics['unique_content_count'][-1]:,}")
    print(f"Training duration: {(metrics['timestamp'][-1] - metrics['timestamp'][0])/3600:.2f} hours")
    
    # Print content diversity metrics
    total_steps = len(metrics['steps'])
    unique_contents = metrics['unique_content_count']
    final_rate = (unique_contents[-1] - unique_contents[-2]) / (metrics['steps'][-1] - metrics['steps'][-2])
    
    print(f"\nContent Diversity Metrics:")
    print(f"Average new examples per step: {unique_contents[-1]/total_steps:.2f}")
    print(f"Final unique content rate: {final_rate:.2f} new examples/step")

if __name__ == "__main__":
    # Load and plot main training metrics
    load_and_plot_metrics("checkpoints/metrics_final.pt")
    
    # Load and plot additional training metrics if available
    # try:
    #    load_and_plot_metrics("checkpoints/metrics_additional.pt")
    # except:
    #     print("No additional training metrics found") 