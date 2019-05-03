import os
import torch

from bc.dataset import Actions


def ppo_model(logdir, device):
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


def _map_numpy_dict_to_tensors(numpy_dict, device):
    for key, value in numpy_dict.items():
        if isinstance(value, (list, tuple)):
            # we expect value to be mean and std
            assert len(value) == 2
            numpy_dict[key] = (torch.tensor(value[0]).float().to(device),
                               torch.tensor(value[1]).float().to(device))
        elif isinstance(value, dict):
            numpy_dict[key] = _map_numpy_dict_to_tensors(value, device)
        else:
            raise NotImplementedError
    return numpy_dict


def bc_checkpoint(args, device):
    if args.checkpoint_path:
        if device.type == 'cpu':
            loaded_dict = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        else:
            loaded_dict = torch.load(args.checkpoint_path)
        args.bc_args = loaded_dict['args']
        print('loaded the BC checkpoint from {}'.format(args.checkpoint_path))
        statistics = _map_numpy_dict_to_tensors(loaded_dict['statistics'], device)
        return args, loaded_dict['model'], statistics
    else:
        if 'Cam' in args.env_name:
            default_bc_args = dict(
                archi='resnet_18',
                mode='features',
                input_dim=3,
                num_frames=3,
                steps_action=4,
                action_space='tool_lin',
                dim_action=4,
                features_dim=512)
            print('did not load a BC checkpoint, using default BC args: {}'.format(default_bc_args))
        else:
            assert args.mime_action_space is not None
            default_bc_args = dict(
                action_space=args.mime_action_space,
                dim_action=Actions.action_space_to_keys(args.mime_action_space)[1],
                num_frames=1)
            print('Using a full state env with BC args: {}'.format(default_bc_args))
        args.bc_args = default_bc_args

        return args, None, None


def _map_state_dict_key(key, num_skills):
    # TODO: reuse for master layers pretraining?
    raise NotImplementedError
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


def policy_from_bc_model(policy, model):
    raise NotImplementedError
    state_dict_renamed = {}
    for key, value in model.net.state_dict().items():
        key_new = _map_state_dict_key(key, policy.base.num_skills)
        state_dict_renamed[key_new] = value

    keys_difference = set(policy.state_dict().keys()) - set(state_dict_renamed)
    if policy.base.action_memory > 0:
        state_dict_renamed = {k: v for k, v in state_dict_renamed.items() if 'actor' not in k}
    # the strict param is for the critics layers only
    assert all(['base.critic' in key for key in keys_difference])
    policy.load_state_dict(state_dict_renamed, strict=False)


def optimizer(optimizer, optimizer_state_dict, device):
    optimizer.load_state_dict(optimizer_state_dict)
    target_device = 'cpu' if device.type == 'cpu' else 'cuda'
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = getattr(v, target_device)()
