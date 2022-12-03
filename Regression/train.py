import torch
import torch.nn as nn
from utils import get_hparams_from_file, load_checkpoint, save_checkpoint
from dataset import build_dataloader
from models import build_model
import numpy as np
import os


def main(config_path):
    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    config = get_hparams_from_file(config_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.train.seed)

    train_loader, stats = build_dataloader(path=config.data.train_path,
                                               mode='train',
                                               batch_size=config.train.batch_size,
                                               device=device)

    dev_loader, stats = build_dataloader(path=config.data.dev_path,
                                        mode='dev',
                                        stats=stats,
                                        batch_size=config.train.batch_size,
                                        device=device)

    predictor, optimizer = build_model(config)
    predictor.to(device)

    predictor.train()

    epoch = 0
    global_step = 0
    early_stop_count = 0

    loss_record = {'train': [], 'dev': []}
    min_mse = 1000

    criterion = nn.MSELoss(reduction='mean')

    while epoch < config.train.epochs:
        counter = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = predictor(x)
            mse_loss = criterion(pred, y)
            mse_loss.backward()
            optimizer.step()

            loss_record['train'].append(mse_loss.detach().cpu().item())

            global_step += 1
            counter += 1

            if global_step % config.train.log_interval == 0:
                progress = counter / len(train_loader) * 100
                log = 'Epoch [{}/{}]: {:.2f}%, step {}, MSE loss: {:.4f}'. \
                    format(epoch, config.train.epochs, progress, global_step, mse_loss)
                print(log)

        predictor.eval()
        total_loss = 0.0
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = predictor(x)
                mse_loss = criterion(pred, y)
            total_loss += mse_loss.detach().cpu().item() * len(x)
        total_loss = total_loss / len(dev_loader.dataset)  # 平均loss
        loss_record['dev'].append(total_loss)

        log = 'Validation at epoch {}, MSE loss: {}'.format(epoch, total_loss)
        print(log)

        if total_loss < min_mse:
            min_mse = total_loss
            save_checkpoint(config.train.save_path, predictor, optimizer, config.train.learning_rate, epoch)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > config.train.early_stop_count:
            break

        epoch += 1

    print('Finished training after {} epochs'.format(epoch))

    train_loss = np.array(loss_record['train'])
    dev_loss = np.array(loss_record['dev'])

    np.save('experiments/train_loss.npy', train_loss)
    np.save('experiments/dev_loss.npy', dev_loss)


if __name__ == '__main__':
    config_path = 'configs/config.json'
    main(config_path)