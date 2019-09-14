import torch
import torch.nn as nn
import torch.nn.functional as F
# define base learner 
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)
    
    def forward(self, x):
        x = x.view(-1, 1)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)

if __name__ == "__main__":
    from generate_data import DataGenerator
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt

    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    model = Regressor()
    data_for_test = DataGenerator(batch=1)
    optimizer = optim.Adam(model.parameters(), lr=0.3)
    loss_fn = nn.MSELoss()
    loss_curve = []
    for i in data_for_test():
        x, y = data_for_test.sample(10)
        plt.scatter(x, y, label='Training data', color='black')
        x, y = to_tensor(x), to_tensor(y)
        for _ in range(10):
            # meta model training
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss_curve.append(loss.item())
            loss.backward()
            optimizer.step()
    
    amp = data_for_test.amp[0]
    phase = data_for_test.phase[0]
    sin_fun = lambda x: amp * np.sin(x + phase)
    x_list = np.linspace(-5, 5, 100)
    real_y_list = [sin_fun(i) for i in x_list]
    fitted_y = model(torch.tensor(x_list, dtype=torch.float32).view(-1, 1))
    fitted_y_list = [i[0] for i in fitted_y.tolist()]
    
    plt.plot(x_list, real_y_list, label='ground truth')
    plt.plot(x_list, fitted_y_list, label='Fitted model')
    plt.legend()
    plt.show()

    plt.plot(loss_curve, label='loss')
    plt.title('loss')
    plt.legend()
    plt.show()
