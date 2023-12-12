import torch
import numpy as np

class LinearDynamicsModel:
    def __init__(self, Mass, Stiffness, Damping, fs, device):
        self.Mass = Mass
        self.Stiffness = Stiffness
        self.Damping = Damping
        self.device = device
        self.dt = 1 / fs

        # Define the system matrix A and input matrix B on the specified device
        self.A = np.block([
            [np.zeros_like(Mass), np.eye(Mass.shape[0])],
            [-np.linalg.inv(Mass) @ Stiffness, -np.linalg.inv(Mass) @ Damping]
        ])

        self.B = np.block([
            np.zeros(Mass.shape),
            np.linalg.inv(Mass)
        ]).T

        self.A = torch.tensor(self.A, dtype=torch.float32, device=device)
        self.B = torch.tensor(self.B, dtype=torch.float32, device=device)
        self.A_inv = torch.inverse(self.A)

        self.A_k = torch.matrix_exp(self.A * self.dt)
        self.B_k = self.A_inv @ (self.A_k - torch.eye(self.A.shape[0], device=device)) @ self.B


    def step(self, x, u):
        x = self.A_k @ x + self.B_k @ u
        return x
    
    def step_model(self, epsilon):
        epsilon = self.B_k @ epsilon
        return epsilon



class NonLinearDynamicsModel:
    def __init__(self, Mass, Stiffness, Damping, fs, device, k3vec, c2vec):
        self.Mass = Mass
        self.Stiffness = Stiffness
        self.Damping = Damping
        self.device = device
        self.dt = 1 / fs
        self.device = device  # The device is passed as an argument
        self.dof = Mass.shape[0]

       # Define the system matrix A and input matrix B on the specified device
        A = np.block([
            [np.zeros_like(Mass), np.eye(Mass.shape[0])],
            [-np.linalg.inv(Mass) @ Stiffness, -np.linalg.inv(Mass) @ Damping]
        ])

        B = np.block([
            np.zeros(Mass.shape),
            np.linalg.inv(Mass)
        ]).T

        k3_A = np.block([
            [np.zeros_like(Mass), np.zeros_like(Mass)],
            [-np.linalg.inv(Mass) @ np.diag(k3vec), np.zeros_like(Mass)]
        ])
        c2_A = np.block([
            [np.zeros_like(Mass), np.zeros_like(Mass)],
            [np.zeros_like(Mass), --np.linalg.inv(Mass) @ np.diag(c2vec)]
        ])


        self.A = torch.tensor(A, dtype=torch.float32, device=device)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.A_k3 = torch.tensor(k3_A, dtype=torch.float32, device=device)
        self.A_c2 = torch.tensor(c2_A, dtype=torch.float32, device=device)

        self.current_state = 0
        for i in range(self.dof):
            if k3vec[i] != 0:
                self.current_state = 1
                break
            elif c2vec[i] != 0:
                self.current_state = 2

        self.print_attributes()

    def print_attributes(self):
        print("Current State:", self.current_state)
        print("A Shape:", self.A.shape if hasattr(self.A, 'shape') else None)
        print("B Shape:", self.B.shape if hasattr(self.B, 'shape') else None)
        print("A_k3 Shape:", self.A_k3.shape if hasattr(self.A_k3, 'shape') else None)
        print("A_c2 Shape:", self.A_c2.shape if hasattr(self.A_c2, 'shape') else None)


    def step(self, x, u):

        if self.current_state == 0:
            k1 = self.dt * (self.A @ x + self.B @ u)
            k2 = self.dt * (self.A @ (x + 0.5 * k1) + self.B @ u)
            k3 = self.dt * (self.A @ (x + 0.5 * k2) + self.B @ u)
            k4 = self.dt * (self.A @ (x + k3) + self.B @ u)
            x_next = x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x_next
        
        elif self.current_state == 1:
            
            k1 = self.dt * (self.A @ x + self.B @ u + self.A_k3 @ (x ** 3))
            k2 = self.dt * (self.A @ (x + 0.5 * k1) + self.B @ u + self.A_k3 @ ((x + 0.5 * k1) ** 3))
            k3 = self.dt * (self.A @ (x + 0.5 * k2) + self.B @ u + self.A_k3 @ ((x + 0.5 * k2) ** 3))
            k4 = self.dt * (self.A @ (x + k3) + self.B @ u + self.A_k3 @ ((x + k3) ** 3))
            x_next = x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x_next
        
        elif self.current_state == 2:
            k1 = self.dt * (self.A @ x + self.B @ u + self.A_c2 @ (x ** 2))
            k2 = self.dt * (self.A @ (x + 0.5 * k1) + self.B @ u + self.A_c2 @ ((x + 0.5 * k1) ** 2))
            k3 = self.dt * (self.A @ (x + 0.5 * k2) + self.B @ u + self.A_c2 @ ((x + 0.5 * k2) ** 2))
            k4 = self.dt * (self.A @ (x + k3) + self.B @ u + self.A_c2 @ ((x + k3) ** 2))
            x_next = x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x_next
        else :
            return None
        
    def step_model(self, epsilon):
        epsilon = self.dt * (self.B @ epsilon)
        return epsilon
