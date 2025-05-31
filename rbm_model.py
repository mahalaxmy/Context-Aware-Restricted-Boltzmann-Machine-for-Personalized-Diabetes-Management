import torch
import torch.nn as nn

class ContextAwareRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, n_context):
        super(ContextAwareRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.context_W = nn.Parameter(torch.randn(n_hidden, n_context) * 0.1)

    def forward(self, v, context):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + torch.matmul(context, self.context_W.t()) + self.h_bias)
        h_sample = torch.bernoulli(h_prob)
        v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
        return v_prob
