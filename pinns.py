#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# seed setting
seed = 0
np.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# initial condition
x0 = 0
k = 0.1
g = 9.8
m = 1
y0 = 2
vx0 = 20
vy0 = 20
t0 = 0
t_end = 3
t_num = 100

t_ic_train = np.array([[t0]])
x_ic_train = np.array([[x0]])
y_ic_train = np.array([[y0]])
vx_ic_train = np.array([[vx0]])
vy_ic_train = np.array([[vy0]])

t_ic_data = torch.tensor(
    t_ic_train, requires_grad=True, dtype=torch.float32).to(device)
x_ic_data = torch.tensor(
    x_ic_train, requires_grad=True, dtype=torch.float32).to(device)
y_ic_data = torch.tensor(
    y_ic_train, requires_grad=True, dtype=torch.float32).to(device)
vx_ic_data = torch.tensor(
    vx_ic_train, requires_grad=True, dtype=torch.float32).to(device)
vy_ic_data = torch.tensor(
    vy_ic_train, requires_grad=True, dtype=torch.float32).to(device)

t = np.linspace(t0, t_end, t_num)
t_train = np.array([[i] for i in t])
t_data = torch.tensor(
    t_train, requires_grad=True, dtype=torch.float32).to(device)

layer_num = int(input('層の数を入力してください：'))
node_num = int(input('ノードの数を入力してください：'))

layer_list = [1] + layer_num*[node_num] + [4]


class linear_layer(nn.Module):
    def __init__(self, inner_neuron, output_neuron):
        super().__init__()
        self.layer = nn.Linear(inner_neuron, output_neuron)

    def forward(self, x):
        x = self.layer(x)
        x = torch.sigmoid(x)
        return x


class Model(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.input_layer = nn.Linear(layer_list[0], layer_list[1])
        self.hidden_layer = self.make_layer(layer_list[1:-1])
        self.output_layer = nn.Linear(layer_list[-2], layer_list[-1])

    def make_layer(self, layer_list):
        layer = []
        for i in range(len(layer_list) - 1):
            linear_block = linear_layer(layer_list[i], layer_list[i+1])
            layer.append(linear_block)
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.sigmoid(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def gradient(outputs, inputs):
    """
    calculate gradient
    """
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs), create_graph=True)


def cal_r2(real_data, pred_data):
    """
    calculate R2
    """
    try:
        ave = np.mean(real_data)
        under = np.sum((real_data-ave)**2)
        upper = np.sum((real_data-pred_data)**2)
        r2 = 1-(upper/under)
    except ZeroDivisionError:
        r2 = -10000
    return r2


model = Model(layer_list)
model.apply(weight_init)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
limit_time = 30

loss_ic_rec = []
loss_pde_rec = []
loss_rec = []
loss_pde1_rec = []
loss_pde2_rec = []
loss_pde3_rec = []
loss_pde4_rec = []

start_time = time.time()

it = 0

print('\n学習開始')
print('-'*50)

while True:
    cal_time = time.time() - start_time

    if cal_time >= limit_time:
        print('-'*50)
        print('学習終了\n')
        print('学習回数：{}回'.format(it))
        print('-'*50)
        break

    output = model(t_data)
    x_pred = output[:, 0:1]
    y_pred = output[:, 1:2]
    vx_pred = output[:, 2:3]
    vy_pred = output[:, 3:4]

    output_ic = model(t_ic_data)
    x_ic_pred = output_ic[:, 0:1]
    y_ic_pred = output_ic[:, 1:2]
    vx_ic_pred = output_ic[:, 2:3]
    vy_ic_pred = output_ic[:, 3:4]

    x_grad = gradient(x_pred, t_data)[0]
    x_t = x_grad[:, :1]

    y_grad = gradient(y_pred, t_data)[0]
    y_t = y_grad[:, :1]

    vx_grad = gradient(vx_pred, t_data)[0]
    vx_t = vx_grad[:, :1]

    vy_grad = gradient(vy_pred, t_data)[0]
    vy_t = vy_grad[:, :1]

    loss_pde_1 = torch.abs(m*vx_t + k*vx_pred).mean()
    loss_pde_2 = torch.abs(m*vy_t + m*g + k*vy_pred).mean()
    loss_pde_3 = torch.abs(x_t - vx_pred).mean()
    loss_pde_4 = torch.abs(y_t - vy_pred).mean()
    loss_pde = loss_pde_1 + loss_pde_2 + loss_pde_3 + loss_pde_4

    loss_ic_1 = torch.abs(x_ic_pred-x_ic_data).mean()
    loss_ic_2 = torch.abs(y_ic_pred-y_ic_data).mean()
    loss_ic_3 = torch.abs(vx_ic_pred-vx_ic_data).mean()
    loss_ic_4 = torch.abs(vy_ic_pred-vy_ic_data).mean()
    loss_ic = loss_ic_1 + loss_ic_2 + loss_ic_3 + loss_ic_4

    loss = loss_ic + loss_pde

    loss_rec.append(loss.item())

    """
    loss_ic_rec.append(loss_ic.item())
    loss_pde_rec.append(loss_pde.item())
    loss_pde1_rec.append(loss_pde_1.item())
    loss_pde2_rec.append(loss_pde_2.item())
    loss_pde3_rec.append(loss_pde_3.item())
    loss_pde4_rec.append(loss_pde_4.item())
    """

    for param in model.parameters():
        param.grad = None
    loss.backward()
    optimizer.step()

    it += 1

    if it % 100 == 0:
        # print('it : {}, loss : {:.2e}'.format(it,loss.item()))
        print('{:.2f}秒'.format(cal_time), end='')
        print(', 学習回数:{}回, 進捗:['.format(it), end='')
        print('#'*int(cal_time), end='')
        print('*'*(limit_time-int(cal_time)), end='')
        print('], {:.0f}%'.format(100*cal_time/limit_time))

x = (m/k)*vx0*(1-np.exp(-(k*t/m)))
y = -m*g*t/k - (m/k)*(vy0+(m*g/k))*np.exp(-k*t/m) + y0 + (m/k)*(vy0+(m*g/k))
vx = vx0 * np.exp(-k*t)
vy = -m*g/k + (vy0+(m*g/k))*np.exp(-k*t/m)

output = model(t_data)
x_pred = output[:, 0:1]
y_pred = output[:, 1:2]
vx_pred = output[:, 2:3]
vy_pred = output[:, 3:4]
x_pred = x_pred.to('cpu').detach().numpy().reshape(100)
y_pred = y_pred.to('cpu').detach().numpy().reshape(100)
vx_pred = vx_pred.to('cpu').detach().numpy().reshape(100)
vy_pred = vy_pred.to('cpu').detach().numpy().reshape(100)

x_r2 = cal_r2(x, x_pred)
y_r2 = cal_r2(y, y_pred)
vx_r2 = cal_r2(vx, vx_pred)
vy_r2 = cal_r2(vy, vy_pred)

ave_r2 = (x_r2 + y_r2 + vx_r2 + vy_r2)/4

if ave_r2 < 0:
    print('精度：0%')
else:
    print('精度：{:.3f}%'.format(100*ave_r2))

plt.plot(x_pred, y_pred, label='pred')
plt.plot(x, y, label='exact', linestyle='dashed')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.show()

# plt.plot(loss_rec)
# plt.yscale('log')
# plt.show()
