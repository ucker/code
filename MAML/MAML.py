import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# from torchviz import make_dot

from Regressor import Regressor
from generate_data import DataGenerator


# copy parameters and create computation graph
def update_params(model, grad_list, lr):
    model_ = Regressor()
    if not model._modules:
        for index, (param_name, _) in enumerate(model.named_parameters()):
            model_._parameters[param_name] = (
                model._parameters[param_name] - lr * grad_list[index]
            )
    else:
        cnt = 0
        for module_name, _ in model.named_children():
            for param_name, _ in model._modules[module_name].named_parameters():
                model_._modules[module_name]._parameters[param_name] = (
                    model._modules[module_name]._parameters[param_name] - lr * grad_list[cnt]
                )
                cnt += 1
    return model_

# update data in model and don't create graph
# like `optimizer.step()`
def update_data(model, grad_list, lr):
    if not model._modules:
        for index, (param_name, _) in enumerate(model.named_parameters()):
            model._parameters[param_name].data += - lr * grad_list[index].data
    else:
        cnt = 0
        for module_name, _ in model.named_children():
            for param_name, _ in model._modules[module_name].named_parameters():
                model._modules[module_name]._parameters[param_name].data += - lr * grad_list[cnt].data
                cnt += 1

to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

if __name__ == "__main__":
    loss_fn = nn.MSELoss()
    model = Regressor()
    # meta_optimizer = optim.SGD(model.parameters(), lr=0.0015)
    meta_optimizer = optim.Adam(model.parameters(), lr=0.01)
    for i in range(100):
        meta_update_loss = 0
        if i % 10 == 0:
            print('iteration {}...'.format(i))
        meta_optimizer.zero_grad()
        dg = DataGenerator(batch=50)
        for _ in dg():
            x, y = dg.sample(100)
            x, y = to_tensor(x), to_tensor(y)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            grad_list = grad(loss, model.parameters(), create_graph=True)
            # The paramters in `meta_model` is not `leaf` any more.
            # We just need to backpropagate these parameters 
            meta_model = update_params(model, grad_list, lr=0.01)
            # sample again to calculate the meta loss
            meta_x, meta_y = dg.sample(10)
            meta_x, meta_y = to_tensor(meta_x), to_tensor(meta_y)
            meta_y_hat = meta_model(meta_x)
            meta_update_loss += loss_fn(meta_y, meta_y_hat)
        # grad_list = grad(meta_update_loss, model.parameters())
        # update_data(model, grad_list, lr=0.01)
        meta_update_loss.backward()
        meta_optimizer.step()
    
    # Comparision
    old_model = Regressor()
    data_for_test = DataGenerator(batch=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer_b = optim.Adam(old_model.parameters(), lr=0.3)
    loss_curve_meta = []
    loss_curve_no_meta = []
    for i in data_for_test():
        x, y = data_for_test.sample(10)
        plt.scatter(x, y, label='Training data', color='black')
        x, y = to_tensor(x), to_tensor(y)
        for _ in range(10):
            # meta model training
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss_curve_meta.append(loss.item())
            loss.backward()
            optimizer.step()
            
            # non-meta model training
            optimizer_b.zero_grad()
            y_hat_ = old_model(x)
            loss_ = loss_fn(y, y_hat_)
            loss_curve_no_meta.append(loss_.item())
            loss_.backward()
            optimizer_b.step()
    
    amp = data_for_test.amp[0]
    phase = data_for_test.phase[0]
    sin_fun = lambda x: amp * np.sin(x + phase)
    x_list = np.linspace(-5, 5, 100)
    real_y_list = [sin_fun(i) for i in x_list]
    fitted_y = model(torch.tensor(x_list, dtype=torch.float32).view(-1, 1))
    fitted_y_ = old_model(torch.tensor(x_list, dtype=torch.float32).view(-1, 1))
    # print(list(fitted_y))
    fitted_y_list = [i[0] for i in fitted_y.tolist()]
    fitted_y_list_ = [i[0] for i in fitted_y_.tolist()]
    # print(x_list)
    # print(fitted_y_list)
    plt.plot(x_list, real_y_list, label='ground truth')
    plt.plot(x_list, fitted_y_list, label='model with MAML')
    plt.plot(x_list, fitted_y_list_, label='model without MAML')
    plt.legend()
    plt.show()
    plt.plot(loss_curve_meta, label='meta model loss')
    plt.plot(loss_curve_no_meta , label='no meta model loss')
    plt.legend()
    plt.show()
