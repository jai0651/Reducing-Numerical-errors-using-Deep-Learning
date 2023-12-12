import torch

class Linear_Spring_Model:
    def __init__(self, m, c, k, t0, tf, fs, device):
        self.m = m
        self.c = c
        self.k = k
        self.t0 = t0
        self.tf = tf
        self.fs = fs
        self.dt = 1 / fs
        self.device = device  # The device is passed as an argument

        # Define the system matrix A and B on the specified device
        self.A = torch.tensor([[0, 1], [-k / m, -c / m]], dtype=torch.float32, device=device)
        self.B = torch.tensor([[0], [1 / m]], dtype=torch.float32, device=device)
        self.A_inv = torch.inverse(self.A)

        self.A_k = torch.matrix_exp(self.A * self.dt)
        self.B_k = self.A_inv @ (self.A_k - torch.eye(2, device=device)) @ self.B

    def step(self, x, u):
        x = self.A_k @ x + self.B_k * u
        return x
    
    def step_model(self, epsilon):
        return self.B_k @ epsilon



class Spring_Model:
    def __init__(self, m, c, k, t0, tf, fs, device, non_linearity):
        self.m = m
        self.c = c
        self.k = k
        self.t0 = t0
        self.tf = tf
        self.fs = fs
        self.dt = 1 / fs
        self.device = device  # The device is passed as an argument

        # Define the system matrix A and B on the specified device
        self.A = torch.tensor([[0, 1], [-k / m, -c / m]], dtype=torch.float32, device=device)
        self.B = torch.tensor([[0], [1 / m]], dtype=torch.float32, device=device)

        self.k3 = non_linearity[0]
        self.order_k3 = non_linearity[1]
        self.c3 = non_linearity[2]
        self.order_c3 = non_linearity[3]

        self.C_k = torch.tensor([[0, 0], [-self.k3 / m, 0]], dtype=torch.float32, device=device)
        self.C_c = torch.tensor([[0, 0], [0, -self.c3 / m]], dtype=torch.float32, device=device)

        if (non_linearity[4] == 0):
          self.C = self.C_k
          self.order = self.order_k3
        else:
          self.C = self.C_c
          self.order = self.order_c3


    def step(self, x, u):

        k1 = self.dt * (self.A @ x + self.B * u + self.C @ (x ** self.order))
        k2 = self.dt * (self.A @ (x + 0.5 * k1) + self.B * u + self.C @ ((x + 0.5 * k1) ** self.order))
        k3 = self.dt * (self.A @ (x + 0.5 * k2) + self.B * u + self.C @ ((x + 0.5 * k2) ** self.order))
        k4 = self.dt * (self.A @ (x + k3) + self.B * u + self.C @ ((x + k3) ** self.order))

        x_next = x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next
    
    
