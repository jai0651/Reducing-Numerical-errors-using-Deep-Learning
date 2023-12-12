import matplotlib.pyplot as plt
import numpy as np

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_losses(losses, window_size):
        # Plot the original losses
    plt.plot(losses, label='Original Loss')
    # Calculate and plot the smoothed losses
    if len(losses) >= window_size:
        smoothed_losses = moving_average(losses, window_size)
        plt.plot(np.arange(window_size-1, len(losses)), smoothed_losses, label='Smoothed Loss (Window=50)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Optimization with Smoothed Loss')
    plt.legend()
    plt.yscale("log")
    plt.show()