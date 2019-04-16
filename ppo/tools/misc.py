import torch
import torch.nn as nn

import os

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


def seed_torch(args):
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)


def try_to_load_model(logdir, device):
    loaded_tuple = None
    for model_name in ('model_current.pt', 'model.pt'):
        try:
            if str(device) == 'cpu':
                loaded_tuple = torch.load(
                    os.path.join(logdir, model_name), map_location=lambda storage, loc: storage)
            else:
                loaded_tuple = torch.load(os.path.join(logdir, model_name))
            print('loaded a policy from {}'.format(os.path.join(logdir, model_name)))
            break
        except Exception as e:
            pass
    return loaded_tuple


# dirty stuff, TODO: refactor later
def map_state_dict_key(key, num_skills):
    # BC training produces names of weights with "module." in the beginning
    if 'fcs' not in key:
        return key.replace('module.', 'base.resnet.')
    fcs_idx = int(key.replace('module.fcs.', '').split('.')[0])
    layer_idx, suffix = key.replace('module.fcs.', '').split('.')[1:]
    layer_idx = int(layer_idx)
    if fcs_idx < num_skills:
        # this is a part of the skills
        key_new = 'base.skills.{}.{}.{}'.format(fcs_idx, layer_idx, suffix)
    else:
        # this is a part of the master
        if layer_idx in {0, 2}:
            # first two fc layers, should go to the hidden actor
            key_new = 'base.actor.{}.{}'.format(layer_idx, suffix)
        else:
            # last layer, should go to the dist
            key_new = 'dist.linear.{}'.format(suffix)
    return key_new


def load_from_checkpoint(policy, path, device):
    if device.type == 'cpu':
        loaded_dict = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        loaded_dict = torch.load(path)
    import pudb; pudb.set_trace()
    model = loaded_dict['model']
    state_dict = model.state_dict()
    state_dict_renamed = {}
    for key, value in state_dict.items():
        key_new = map_state_dict_key(key, policy.base.num_skills)
        state_dict_renamed[key_new] = value

    keys_difference = set(policy.state_dict().keys()) - set(state_dict_renamed)
    # the strict param is for the critics layers only
    assert all(['base.critic' in key for key in keys_difference])
    # policy.load_state_dict(state_dict_renamed, strict=False)
    state_dict = {}
    # TODO: do not remove the actor layers
    for key in state_dict_renamed.keys():
        if 'actor' not in key:
            state_dict[key] = state_dict_renamed[key]
    policy.load_state_dict(state_dict, strict=False)
    print('loaded the BC checkpoint from {}'.format(path))
    return loaded_dict['statistics']


def load_optimizer(optimizer, optimizer_state_dict, device):
    optimizer.load_state_dict(optimizer_state_dict)
    target_device = 'cpu' if device.type == 'cpu' else 'cuda'
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = getattr(v, target_device)()


def dict_to_tensor(dictionary):
    ''' Function to make a tensor out of a dictionary where the keys are env_idxs '''
    if dictionary is None:
        return dictionary, None
    tensor_list = []
    for key in sorted(dictionary.keys()):
        tensor_list.append(dictionary[key])
    return torch.stack(tensor_list), sorted(dictionary.keys())


def tensor_to_dict(tensor, keys=None):
    ''' Function to make a dictionary out of a tensor where the keys are env_idxs '''
    if tensor is None:
        return tensor, None
    if keys is None:
        keys = range(tensor.shape[0])
    dictionary = {}
    for idx, key in enumerate(keys):
        dictionary[key] = tensor[idx]
    return dictionary, keys


def pudb():
    try:
        from pudb.remote import set_trace
        set_trace()
    except:
        pass

