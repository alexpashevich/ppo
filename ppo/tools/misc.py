import torch
import random
import torch.nn as nn
import numpy as np


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def seed_exp(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)


def dict_to_tensor(dictionary):
    ''' Function to make a tensor out of a dictionary where the keys are env_idxs '''
    if dictionary is None:
        return dictionary, None
    tensor_list = []
    for key in sorted(dictionary.keys()):
        tensor_list.append(dictionary[key])
    tensor = torch.stack(tensor_list)
    # assert check
    for tensor_idx, env_idx in enumerate(sorted(dictionary.keys())):
        assert tensor[tensor_idx].float().mean() == dictionary[env_idx].float().mean()
    return tensor, sorted(dictionary.keys())


def tensor_to_dict(tensor, keys=None):
    ''' Function to make a dictionary out of a tensor where the keys are env_idxs '''
    if tensor is None:
        return tensor, None
    if keys is None:
        keys = range(tensor.shape[0])
    dictionary = {}
    for idx, key in enumerate(keys):
        dictionary[key] = tensor[idx]
    # assert check
    for tensor_idx, env_idx in enumerate(sorted(keys)):
        assert tensor[tensor_idx].float().mean() == dictionary[env_idx].float().mean()
    return dictionary, keys


def pudb():
    try:
        from pudb.remote import set_trace
        set_trace()
    except:
        pass
