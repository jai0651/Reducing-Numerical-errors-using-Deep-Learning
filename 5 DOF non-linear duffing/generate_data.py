import torch
from scipy.signal import butter, lfilter

def generate_date(simulator, u):
    N = u.shape[0]
    yi = torch.zeros(10, 1).to(simulator.device)
    x = torch.zeros((10, N), dtype=torch.float32, device=simulator.device)
    for m in range(N - 1):
        u_m = u[m:m+1, :].T
        yi = simulator.step(yi, u_m)
        x[:, m + 1: m + 2] = yi
    return x


def generate_u(sig_u, nt, fs, device):
    # External input excitation
    if sig_u > 0:
        utmp = sig_u * torch.randn(nt, 1)
        cutoff_freq = 20  # Hz
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        
        # Apply the filter to the input signal
        u = torch.tensor(lfilter(b, a, utmp.squeeze().numpy()), dtype=torch.float32)
        
        u = u.to(device)
    else:
        u = torch.tensor([])

    return u

def generate_u_sinusoidal(sig_u, freq, tspan, nt, device):
    u_sinusoidal = 0.2 * torch.sin(2 * torch.pi * freq * tspan)
    u = u_sinusoidal + sig_u * torch.randn_like(u_sinusoidal)
    return u_sinusoidal

