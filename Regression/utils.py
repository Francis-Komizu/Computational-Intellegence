import json
import os
import torch


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:  # resume training
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict(checkpoint_dict['model'])
    print('Loaded model and optimizer state at epoch {} from {}'.format(epoch, checkpoint_path))

    return model, optimizer, learning_rate, epoch


def save_checkpoint(checkpoint_path, model, optimizer, learning_rate, epoch):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'epoch': epoch
    }

    torch.save(state_dict, checkpoint_path)
    print('Saved model and optimizer state at epoch {} to {}'.format(epoch, checkpoint_path))


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
