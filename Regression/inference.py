import torch
import os
import csv
from utils import load_checkpoint, get_hparams_from_file
from models import Predictor
from dataset import build_dataloader


def save_result(preds, save_path):
    with open(save_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['MEDV'])
        for i, p in enumerate(preds):
            writer.writerow([p])


def inference(config_path, checkpoint_path, save_path, stats):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = get_hparams_from_file(config_path)

    test_loader, stats = build_dataloader(path=config.data.test_path,
                                          mode='test',
                                          stats=stats,
                                          batch_size=1,
                                          device=device)

    predictor = Predictor(input_dim=config.model.input_dim)
    predictor, _, _, _ = load_checkpoint(checkpoint_path, predictor)
    predictor.eval()

    preds = []

    for x in test_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = predictor(x)
            pred = torch.exp(pred)  # 训练时取了log
            preds.append(pred)

    preds = torch.cat(preds, dim=0).numpy()

    save_result(preds, save_path)


if __name__ == '__main__':
    config_path = 'configs/config.json'
    checkpoint_path = 'models/checkpoint.pth'
    test_path = 'data/test_set.csv'
    save_path = 'experiments/preds.csv'

    _, stats = build_dataloader('data/train.csv',
                                batch_size=16,
                                mode='train')

    inference(config_path, checkpoint_path, save_path, stats)
