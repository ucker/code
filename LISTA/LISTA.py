# LISTA Method
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data
from training_data import sparse_coding

# Learnable lambd
# source:https://discuss.pytorch.org/t/how-can-i-make-the-lambda-trainable-in-softshrink-function/8733/2
def softshrink(x, lambd):
    mask1 = x > lambd
    mask2 = x < -lambd
    out = torch.zeros_like(x)
    out += mask1.float() * -lambd + mask1.float() * x
    out += mask2.float() * lambd + mask2.float() * x
    return out

class Net(nn.Module):
    def __init__(self, n, m, weights, L):
        '''
        Args:
            n : data dimension
            m : sparse code dimension
            weights : optimal dictionary
            L : see ISTA
        '''
        super(Net, self).__init__()
        self.n = n
        self.m = m
        self.W = nn.Parameter(torch.Tensor(m, n))
        self.S = nn.Parameter(torch.Tensor(m, m))
        self.theta = nn.Parameter(torch.Tensor(m))
        self.reset_parameters(weights, L)
    
    def reset_parameters(self, weights, L):
        # weights initialization
        self.W.data.set_(torch.from_numpy(weights.T/L).float())
        S = np.eye(self.m) - np.dot(weights.T, weights) / L
        self.S.data.set_(torch.from_numpy(S).float())
        self.theta.data.set_(torch.Tensor([0.5/L for i in range(self.m)]))

    def forward(self, x):
        z = softshrink(F.linear(x, self.W), self.theta)
        for i in range(1):
            z = softshrink(F.linear(x, self.W) + F.linear(z, self.S), self.theta)
        return z

def train(model, train_loader, optimizer, epoch):
    model.train()
    loss_fn = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def predict_sparse_code(model, test_loader):
    model.eval()
    correct = 0
    output_res = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data).numpy().reshape(-1)
            output_res.append(output)
    return np.array(output_res)



def solve():
    '''
    Returns:
        new_sparse_code : sparse code of test data
        test_data : test data and its label
    '''
    torch.manual_seed(1)
    print("Getting data for LISTA...")
    train_data, sparse_code, opti_Wd, test_data = sparse_coding()
    # get L (larger than  the eigenvalues of W'*W)
    if opti_Wd.shape[1] < 500:
        # eigenvalues of W'*W aren't difficult to calculate
        ss = np.max(np.linalg.eigvals(np.dot(opti_Wd.T, opti_Wd)))
        L = (ss.imag**2 + ss.real**2)**0.5
    else:
        # assign a 'big' value
        L = 1000
    # make pytorch train_loader and test_loader
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data[0]).float(), torch.from_numpy(sparse_code).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data[0]).float(), torch.from_numpy(test_data[1]).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = Net(opti_Wd.shape[0], opti_Wd.shape[1], opti_Wd, L)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print("Training LISTA...")
    for epoch in range(1, 20 + 1):
        train(model, train_loader, optimizer, epoch)
        # get sparse code of test data
    new_sparse_code = predict_sparse_code(model, test_loader)
    return (new_sparse_code, test_data)
