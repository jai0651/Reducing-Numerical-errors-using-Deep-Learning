import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, device):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Create a list to store the layers, including hidden layers
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]).to(device))
        layers.append(activation)

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]).to(device))
            layers.append(activation)

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size).to(device))

        # Define the neural network using Sequential
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.net(x)

class DifferentialSolver:
    def __init__(self, model, optimizer, criterion=nn.MSELoss()):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, simulation, target_solution, u, max_epochs=100, msteps=2, batch_size=1000):
        if max_epochs > 100000:
            display_at = 5000
        elif max_epochs > 10000:
            display_at = 2000
        else:
            display_at = max(1, max_epochs / 100)
        losses = []
        N = target_solution.shape[0]
        feasible_range = N - msteps - 1
        std_data = torch.std(target_solution, axis = 0).T    #torch.Size([2, 5])

        for sim in range(max_epochs):
            self.optimizer.zero_grad()
            loss = 0

            for batch in range(batch_size):
                r = np.random.randint(feasible_range)
                yi = target_solution[r, :].T # torch.Size([2, 5])

                for m in range(msteps):
                    u_m = u[r + m: r + m + 1, :] # torch.Size([1, 5])
                    yi = simulation.step(yi, u_m)
                    input = torch.cat([u_m, yi], axis=0).T  # Concatenate k and yi.T
                    dy_i = self.model(input).T # torch.Size([2, 5])
                    yi = yi + dy_i
                    loss += self.criterion(yi.div(std_data) , target_solution[r + m + 1, :].T.div(std_data)) / msteps

            loss /= batch_size
            loss.backward(retain_graph=True)
            self.optimizer.step()
            losses.append(loss.item())

            if sim % display_at == 0:
                print(sim, "Training Loss:", loss.item())

        return losses