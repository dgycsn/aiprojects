"""
Parity check: input is a sequence of bits (e.g. [1, 0, 1, 1, 0])
output True if the number of 1s in the sequence is even, False if odd.

input dimension = 1, since network sees 1 new bit at a time

output dimension = 1, since output is only true-false

hidden dimension -> you decide yourself
in practice 4-8, but in my case it can simply be 1 as a boolean

equations:
    ht = tanh(W_xh * xt + W_hh * h_(t-1) + bh)
    yt = W_hy * ht + b_y
"""
import numpy as np
# import math
import matplotlib.pyplot as plt

def state(x, h):
    term1 = np.dot(w_xh , x)
    term2 = np.dot(w_hh , h)
    return np.tanh(term1 + term2 + b_h)

def output(h):
    return np.dot(w_hy,h) + b_y

def forward_pass(input_array):
    # assume np.array([input_array[t]])
    input_size = len(input_array[:,0])
    
    state_array = np.zeros((input_size+1, hdim))
    output_array = np.zeros((input_size, ydim))
    
    for t in range(input_size):
        state_array[t+1,:] = state(input_array[t,:], state_array[t,:])
        output_array[t,:] = output(state_array[t+1,:])
    return state_array, output_array

#%%
# simple hand verify 
# input_array = np.array([[1],[1]])
# print(forward_pass(input_array))

#%%

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(label, y):
    # assume scalar
    return -(label * np.log(y) + (1-label) * np.log(1-y))

"""
With sigmoid + binary cross-entropy, there's a well-known simplification
 dL / dy_t = prediction_t - true_label_t
"""

learning_rate = 0.001

# input_array = np.transpose(np.array([[0,1,1,0,0,0,1]]))
# label = [1,0,1,1,1,1,0]
def bptt(input_array,label):
    global w_xh, w_hh, w_hy, b_h, b_y
    input_size = len(input_array[:,0])
    state_array, output_array = forward_pass(input_array)
    
    p_hat = sigmoid(output_array[-1])
    dL_dy = p_hat - label[-1]
    
    state_gradient = np.dot(np.transpose(w_hy), dL_dy)
    
    dW_xh = np.zeros_like(w_xh)
    dW_hh = np.zeros_like(w_hh)
    dB_h = np.zeros_like(b_h)
    
    for t in range(input_size):
        h_t    = state_array[input_size - t]
        h_prev = state_array[input_size - t - 1]
        x_t    = input_array[input_size - t - 1]
    
        dL_dz = np.multiply(state_gradient, 1 - h_t**2)
        dW_hh += np.outer(dL_dz, h_prev)
        dW_xh += np.outer(dL_dz, x_t)
        dB_h += dL_dz
        state_gradient = np.dot(np.transpose(w_hh), dL_dz)
    
    dW_hy = np.outer(dL_dy, state_array[-1,:])
    w_hy -= dW_hy * learning_rate
    b_y -= dL_dy * learning_rate
    w_xh -= learning_rate * dW_xh
    w_hh -= learning_rate * dW_hh
    b_h  -= learning_rate * dB_h
    

#%%
def create_input(x=49):
    size = int(np.random.rand()*x) + 1
    ones_prob = np.random.rand()
    input_array = np.zeros(size)
    label = np.zeros(size)
    even = True
    for i in range(size):
        if np.random.rand() < ones_prob:
            input_array[i] = 1
            even = not even
        label[i] = int(even)
    return np.transpose(np.array([input_array])), label
#%%

xdim = 1
ydim = 1
hdim = 2

w_xh = np.random.rand(hdim,xdim)
w_hh = np.random.rand(hdim,hdim)
w_hy = np.random.rand(ydim,hdim)
b_h = np.random.rand(hdim,)
b_y = np.random.rand(ydim,)

N = 5000
loss_over_epoch = np.zeros(N)
for epoch in range(N):
    x, y = create_input(); 
    _, prediction = forward_pass(x)
    loss_over_epoch[epoch] = loss(y[-1], sigmoid(prediction[-1,0]))
    bptt(x, y)
    
plt.plot(np.linspace(1, N,N), loss_over_epoch)

"""
notes:
    - for size 50 average loss gets stuck at around 0.7 = -ln(0.5) --> predicting50/50
    - same for size 5
    - size 1-2 --> actually drops smoothly to 0 after 500 iterations
"""

#%%
import torch
import torch.nn as nn

xdim = 1
ydim = 1
hdim = 2


cell = nn.RNNCell(input_size=xdim, hidden_size=hdim, nonlinearity='tanh')
linear = nn.Linear(hdim, ydim)

# print("cell.weight_ih:", cell.weight_ih.shape)   # expect (hdim, xdim)
# print("cell.weight_hh:", cell.weight_hh.shape)   # expect (hdim, hdim)
# print("cell.bias_ih:  ", cell.bias_ih.shape)      # expect (hdim,)
# print("cell.bias_hh:  ", cell.bias_hh.shape)      # expect (hdim,)
# print("linear.weight: ", linear.weight.shape)     # expect (ydim, hdim)
# print("linear.bias:   ", linear.bias.shape)       # expect (ydim,)

x_np, y_np = create_input()

x = torch.tensor(x_np, dtype=torch.float32)   #convert numpy array to torch.tensor so autograd can track operations on it
target = torch.tensor([y_np[-1]], dtype=torch.float32)  # convert output with correct shape

h = torch.zeros(1, hdim)   # note: batch dimension of 1 required by RNNCell

for t in range(x.shape[0]):
    x_t = x[t].unsqueeze(0)      # shape (1, xdim) — RNNCell wants a batch dim
    h = cell(x_t, h)

y = linear(h)   # shape (1, ydim)
print(y)

loss_fn = nn.BCEWithLogitsLoss()
target = target.view(1, 1)   # or just build it as torch.tensor([[y_np[-1]]]) above
loss = loss_fn(y, target)
print(loss)

loss.backward()

print(cell.weight_ih.grad)   # gradient for W_xh
print(cell.weight_hh.grad)   # gradient for W_hh
print(linear.weight.grad)    # gradient for W_hy
print(linear.bias.grad)      # gradient for b_y
print(cell.bias_ih.grad + cell.bias_hh.grad)  # combined gradient for b_h
#%%
"""
batching helps
"""
import torch.optim as optim

xdim = 1
ydim = 1
hdim = 5

cell = nn.RNNCell(input_size=xdim, hidden_size=hdim, nonlinearity='tanh')
linear = nn.Linear(in_features=hdim, out_features=ydim)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(list(cell.parameters()) + list(linear.parameters()), lr=0.1)

N = 2000
loss_over_epoch = np.zeros(N)

batch_size = 100
for epoch in range(N):
    optimizer.zero_grad()
    batch_loss = 0
    for _ in range(batch_size):
        x_np, y_np = create_input(5)
        x = torch.tensor(x_np, dtype=torch.float32)
        target = torch.tensor([[y_np[-1]]], dtype=torch.float32)
        h = torch.zeros(1, hdim)
        for t in range(x.shape[0]):
            h = cell(x[t].unsqueeze(0), h)
        y = linear(h)
        batch_loss += loss_fn(y, target)
    batch_loss = batch_loss / batch_size
    batch_loss.backward()
    optimizer.step()

    loss_over_epoch[epoch] = batch_loss.item()

plt.plot(loss_over_epoch)


#%%

def test_accuracy(n_test=200, fixed_len=None):
    correct = 0
    lengths = []
    with torch.no_grad():
        for _ in range(n_test):
            x_np, y_np = create_input(5000)
            if fixed_len is not None:
                x_np, y_np = x_np[:fixed_len], y_np[:fixed_len]
            x = torch.tensor(x_np, dtype=torch.float32)
            h = torch.zeros(1, hdim)
            for t in range(x.shape[0]):
                h = cell(x[t].unsqueeze(0), h)
            y = linear(h)
            pred = (torch.sigmoid(y).item() > 0.5)
            actual = bool(y_np[-1])
            correct += (pred == actual)
            lengths.append(len(x_np))
    print(f"accuracy: {correct/n_test:.2f}")

test_accuracy()
