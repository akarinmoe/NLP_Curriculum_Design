import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# class LSTMCell(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(LSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
#         self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

#     def forward(self, x, hidden):
#         h_prev, c_prev = hidden

#         # Linear transformations
#         combined = self.i2h(x) + self.h2h(h_prev)
#         i_gate, f_gate, c_gate, o_gate = torch.split(combined, self.hidden_size, dim=1)
        
#         # Gate activations
#         i_gate = torch.sigmoid(i_gate)
#         f_gate = torch.sigmoid(f_gate)
#         c_gate = torch.tanh(c_gate)
#         o_gate = torch.sigmoid(o_gate)
        
#         # Cell state and hidden state updates
#         c_new = f_gate * c_prev + i_gate * c_gate
#         h_new = o_gate * torch.tanh(c_new)
        
#         # Save gates and cell states for backpropagation
#         self.saved_tensors = (x, h_prev, c_prev, i_gate, f_gate, c_gate, o_gate, c_new, h_new)
        
#         return h_new, c_new

#     def backward(self, grad_h, grad_c):
#         """
#         grad_h: Gradient of the loss with respect to the next hidden state.
#         grad_c: Gradient of the loss with respect to the next cell state.
#         """
#         # Retrieve saved tensors from the forward pass
#         x, h_prev, c_prev, i_gate, f_gate, c_gate, o_gate, c_new, h_new = self.saved_tensors
        
#         # Gradient of output gate
#         d_o_gate = grad_h * torch.tanh(c_new)
#         d_o_gate *= o_gate * (1 - o_gate)  # Sigmoid derivative
        
#         # Gradient of cell state
#         d_c_new = grad_h * o_gate * (1 - torch.tanh(c_new) ** 2) + grad_c
#         d_f_gate = d_c_new * c_prev
#         d_i_gate = d_c_new * c_gate
#         d_c_gate = d_c_new * i_gate
        
#         # Gate derivatives
#         d_f_gate *= f_gate * (1 - f_gate)  # Sigmoid derivative
#         d_i_gate *= i_gate * (1 - i_gate)  # Sigmoid derivative
#         d_c_gate *= (1 - c_gate ** 2)      # Tanh derivative

#         # Stack gradients to match combined linear layer output
#         grad_combined = torch.cat((d_i_gate, d_f_gate, d_c_gate, d_o_gate), dim=1)

#         # Gradients with respect to inputs and weights
#         grad_x = self.i2h.weight.T @ grad_combined.T
#         grad_h_prev = self.h2h.weight.T @ grad_combined.T
#         grad_x = grad_x.T
#         grad_h_prev = grad_h_prev.T
        
#         # Update the weights
#         self.i2h.weight.grad = grad_combined.T @ x
#         self.h2h.weight.grad = grad_combined.T @ h_prev

#         return grad_x, grad_h_prev, d_c_new



# class MYLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(MYLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.cells = nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

#     def forward(self, x, hidden=None):
#         batch_size, seq_len, _ = x.size()
        
#         if hidden is None:
#             h_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
#             c_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
#             hidden = (h_zeros, c_zeros)
        
#         h, c = hidden
#         layer_out = x
#         outputs = []

#         for t in range(seq_len):
#             x_t = layer_out[:, t, :]
#             h_out, c_out = [], []
            
#             for layer in range(self.num_layers):
#                 h_next, c_next = self.cells[layer](x_t, (h[layer], c[layer]))
#                 x_t = h_next
#                 h_out.append(h_next)
#                 c_out.append(c_next)
                
#             outputs.append(h_next)
#             h, c = torch.stack(h_out), torch.stack(c_out)
        
#         return torch.stack(outputs, dim=1), (h, c)

#     def backward(self, grad_output, hidden_grad=None):
#         """
#         grad_output: Gradient of the loss with respect to the LSTM output.
#         hidden_grad: Gradient of the hidden states from the next layer or time step.
#         """
#         if hidden_grad is None:
#             h_grad = torch.zeros_like(self.hidden_size)
#             c_grad = torch.zeros_like(self.hidden_size)
#         else:
#             h_grad, c_grad = hidden_grad

#         for t in reversed(range(len(grad_output))):
#             grad_out = grad_output[t]
#             for layer in reversed(range(self.num_layers)):
#                 grad_out, h_grad, c_grad = self.cells[layer].backward(grad_out + h_grad, c_grad)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize i2h and h2h layers
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=True)  # input_size -> 4 * hidden_size
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)  # hidden_size -> 4 * hidden_size

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # Ensure x matches the input size
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected x to have input_size={self.input_size}, but got {x.size(1)}")

        # Linear transformations
        combined = self.i2h(x) + self.h2h(h_prev)
        i_gate, f_gate, c_gate, o_gate = torch.split(combined, self.hidden_size, dim=1)
        
        # Gate activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)
        
        # Cell state and hidden state updates
        c_new = f_gate * c_prev + i_gate * c_gate
        h_new = o_gate * torch.tanh(c_new)
        
        # Save gates and cell states for backpropagation
        self.saved_tensors = (x, h_prev, c_prev, i_gate, f_gate, c_gate, o_gate, c_new, h_new)
        
        return h_new, c_new


class MYLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(MYLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize LSTM cells, first layer uses input_size, others use hidden_size
        self.cells = nn.ModuleList(
            [LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, input_dim = x.size()
        
        # Ensure x matches the expected input size
        if input_dim != self.cells[0].input_size:
            raise ValueError(f"Expected input dimension {self.cells[0].input_size}, but got {input_dim}")

        # Initialize hidden states if not provided
        if hidden is None:
            h_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
            c_zeros = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
            hidden = (h_zeros, c_zeros)
        
        h, c = hidden
        outputs = []

        # Iterate over sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_out, c_out = [], []
            
            # Pass through each layer of the LSTM
            for layer in range(self.num_layers):
                h_next, c_next = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h_next  # Pass output to next layer
                h_out.append(h_next)
                c_out.append(c_next)
                
            # Save output for each time step
            outputs.append(h_next)
            h, c = torch.stack(h_out), torch.stack(c_out)
        
        return torch.stack(outputs, dim=1), (h, c)

    def backward(self, grad_output, hidden_grad=None):
        """
        grad_output: Gradient of the loss with respect to the LSTM output.
        hidden_grad: Gradient of the hidden states from the next layer or time step.
        """
        if hidden_grad is None:
            h_grad = torch.zeros_like(self.hidden_size)
            c_grad = torch.zeros_like(self.hidden_size)
        else:
            h_grad, c_grad = hidden_grad

        # Backprop through time for each time step
        for t in reversed(range(len(grad_output))):
            grad_out = grad_output[t]
            for layer in reversed(range(self.num_layers)):
                grad_out, h_grad, c_grad = self.cells[layer].backward(grad_out + h_grad, c_grad)
