import torch
import matplotlib.pyplot as plt
import numpy as np

i = 0

def plot_hist(name, calib, percentile=99.99):
    plt.clf()
    hist_file = f'adc_hist_{name}.pt'

    hist = calib._calib_hist
    x_values = calib._calib_bin_edges

    hist = hist.cpu().numpy()
    x_values = x_values.cpu().numpy()

    total = hist.sum()
    cdf = np.cumsum(hist / total)
    idx = np.searchsorted(cdf, percentile / 100)
    calib_amax = x_values[idx]

    fig, ax1 = plt.subplots()

    # Plot the histogram
    ax1.bar(x_values[:-1], hist)
    ax1.set_xlabel('ADC Output')
    ax1.set_ylabel('Frequency')
    ax1.tick_params('y', colors='blue')

    # Get the largest frequency and the 95th percentile
    largest_frequency = np.max(hist)
    percentile_95 = np.percentile(hist, 99.95)  

    # Set the y limit to the 95th percentile
    ax1.set_ylim([0, percentile_95])

    # Create a secondary y-axis for the CDF
    ax2 = ax1.twinx()
    ax2.plot(x_values[:-1], cdf, color='red')
    ax2.set_ylabel('Cumulative Distribution Function (CDF)', color='red')
    ax2.tick_params('y', colors='red')

    # Add dotted lines for the 99.99% point
    percentile_9999 = np.percentile(cdf, percentile)
    ax2.axhline(y=percentile_9999, color='black', linestyle='--', label=f'{percentile}% Point')
    ax1.axvline(x=calib_amax, color='black', linestyle='--')

    # Annotate the plot with the value of the largest frequency
    plt.annotate(f'ADC Output = 0 Frequency: {largest_frequency:.2e}', xy=(0.4, 0.7), xycoords='axes fraction')

    # Calculate the output sparsity
    output_sparsity = (hist == 0).mean() * 100

    # Annotate the plot with the output sparsity
    plt.annotate(f'Output sparsity: {output_sparsity:.2f}%', xy=(0.5, 0.6), xycoords='axes fraction')
    plt.annotate(f'amax: {calib_amax:.0f}', xy=(0.4, 0.3), xycoords='axes fraction')    

    # Set plot title
    plt.title(f'{name} ADC Output')

    # Show the plot
    plt.show()
    plt.savefig(f'{name}.png')
