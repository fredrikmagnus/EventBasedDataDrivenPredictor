import numpy as np
import matplotlib.pyplot as plt


def compare_predictions(time, x, predictions):
    n_inputs = x.shape[0]
    
    # Standard color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Top subplot: predictions over time
    for i in range(n_inputs):
        ax1.plot(time, predictions[i, :], label=f'$\\hat{{x}}_{{{i+1}}}$', linewidth=2, color=colors[i])

    # ax1.plot(time, np.array(test), linestyle='-', color='black', label='Test Statistic', linewidth=2)
    ax1.set_ylabel('Prediction')
    ax1.set_title('Next-event Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: event plot of original spikes
    for i in range(n_inputs):
        spike_times = time[x[i, :] == 1]
        if len(spike_times) > 0:
            ax2.eventplot(spike_times, lineoffsets=i, linewidths=2, 
                        colors=colors[i], label=f'x{i+1}')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Input Channel')
    ax2.set_title('Observed Spikes')
    ax2.set_yticks(range(n_inputs))
    ax2.set_yticklabels([f'x{i+1}' for i in range(n_inputs)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("SpikePredictionsAndEvents.pdf")
    plt.show()  

def plot_covariances(time, Covs, CrossCovs):
    # Plot all values of K over time
    n_inputs = Covs.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.step(time, Covs[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Covariance Values')
    plt.title(r'$\text{Cov}(z_k^+, z_k^+)$ Matrix Elements Over Time')
    plt.legend()
    plt.grid()


    # Plot all values of K_causal over time
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.step(time, CrossCovs[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Cross-Covariance Values')
    plt.title(r'$\text{Cov}(x_k, z_k^-)$ Matrix Elements Over Time')
    plt.legend()
    plt.grid()
    plt.show()  

def plot_traces(time, traces):
    n_inputs = traces.shape[0]
    plt.figure(figsize=(12, 6))
    for i in range(n_inputs):
        plt.plot(time, traces[i, :], label=f'x{i+1}')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Synaptic Traces over Time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_gains(time, PredictionGains):
    n_inputs = PredictionGains.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.step(time, PredictionGains[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Prediction Gain Values')
    plt.title('Evolution of Prediction Gain Matrix Elements Over Time')
    plt.legend()
    plt.grid()
    plt.show()