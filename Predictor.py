import numpy as np

class Predictor:
    def __init__(self, n_inputs, gamma_weights, tau_decay, lambda_ridge, dt, estimate_mean=False):
        self.n_inputs = n_inputs
        self.gamma_weights = gamma_weights
        self.tau_decay = tau_decay
        self.lambda_ridge = lambda_ridge
        self.dt = dt
        self.decay = np.exp(-dt/self.tau_decay) # Per time-step trace decay

        self.x = np.zeros(n_inputs) # Trace vector
        self.mean = np.zeros(n_inputs) # Event-based mean
        self.z = np.zeros(n_inputs) # Centered traces

        self.Cov = np.zeros((n_inputs, n_inputs))
        self.CrossCov = np.zeros((n_inputs, n_inputs))
        self.P = self.lambda_ridge * np.eye(n_inputs)

        self.estimate_mean = estimate_mean

    def update(self, x_in:np.ndarray):

        # 1) Decay traces:
        self.x *= self.decay

        # 2) Update mean based on pre-spike traces
        if self.estimate_mean and x_in.sum() > 0:
            self.mean = self.gamma_weights*self.mean + (1-self.gamma_weights)*self.x

        # 3) Update traces with incoming spikes
        z_pre = self.x - self.mean
        z_post = z_pre + x_in * (1-self.decay)/self.dt
        self.x += x_in * (1-self.decay)/self.dt

        # 4) Update covariance estimates and prediction weights when there is an input spike
        if x_in.sum() > 0:
            self.CrossCov = self.gamma_weights*self.CrossCov + (1-self.gamma_weights)*np.outer(x_in, z_pre)
            self.Cov = self.gamma_weights*self.Cov + (1-self.gamma_weights)*(np.outer(z_post, z_post) + self.lambda_ridge*np.eye(self.n_inputs))
            self.P = self.CrossCov @ np.linalg.inv(self.Cov)

        self.z = z_post

    def predict(self):
        return self.P @ self.z
    

def spike_signal(t, period, phase, randomize=0):
    """
    Generate periodic spike signal.
    
    Args:
        t: time array
        period: spike period
        phase: phase offset
        randomize: add small random jitter to spike times. Uniform in [-randomize, randomize]
    
    Returns:
        Binary array with spikes at closest indices to theoretical spike times
    """
    spikes = np.zeros_like(t, dtype=int)
    
    # Generate theoretical spike times
    max_time = t[-1]
    n_spikes = int((max_time - phase) / period) + 1
    theoretical_spike_times = phase + np.arange(n_spikes) * period
    
    # Only keep spike times within the time range
    theoretical_spike_times = theoretical_spike_times[
        (theoretical_spike_times >= t[0]) & (theoretical_spike_times <= max_time)
    ]
    
    # Find closest indices for each theoretical spike time
    for spike_time in theoretical_spike_times:
        if randomize:
            jitter = np.random.uniform(-randomize, randomize)
            spike_time += jitter
        closest_idx = np.argmin(np.abs(t - spike_time))
        spikes[closest_idx] = 1
    
    return spikes