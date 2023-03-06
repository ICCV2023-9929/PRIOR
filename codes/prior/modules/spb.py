import numpy as np
import torch
from torch import nn, tensor
from torch.autograd import Function, Variable
import torch.nn.functional as F
from math import log


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    #print(logits[0],"a")
    #print(len(argmax_acs),argmax_acs[0])
    if eps == 0.0:
        return argmax_acs

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        #print(y_hard[0], "random")
        y = (y_hard - y).detach() + y
    return y

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gradient_softmax(logits, temperature=1.0, hard=False):
    y = F.softmax(logits / temperature, dim=1)
    if hard:
        y_hard = onehot_from_logits(y)
        #print(y_hard[0], "random")
        y = (y_hard - y).detach() + y
    return y


class SPB(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, temp=0.9):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embed = nn.Embedding(n_embed, dim)
        self.temp = temp

    def set_temp(self, epoch, max_epoch):
        return # abandom
        
    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        logits = flatten @ self.embed.weight.T
        self.gt_logits = logits
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temp, dim=1, hard=False)
        output = soft_one_hot @ self.embed.weight
        embed_ind = soft_one_hot.argmax(1)
        recon_loss = (output - flatten).abs().mean()
        loss = recon_loss 
        self.logits = logits
        return output, loss, embed_ind

    @ torch.no_grad()
    def query(self, input):
        flatten = input.reshape(-1, self.dim)
        logits = flatten @ self.embed.weight.T
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temp, dim=1, hard=False)
        output = soft_one_hot @ self.embed.weight
        recon_loss = (output - flatten).abs().mean()
        embed_ind = soft_one_hot.argmax(1)
        return output, recon_loss, embed_ind


    def cal_loss(self, input):
        flatten = input.reshape(-1, self.dim)
        logits = flatten @ self.embed.weight.T
        log_gt_logits = F.log_softmax(self.gt_logits, dim=-1)
        log_logits = F.log_softmax(logits, dim=-1)
        kl_div = F.kl_div(log_logits, log_gt_logits, None, None, 'batchmean', log_target = True)
        return kl_div
    

   




