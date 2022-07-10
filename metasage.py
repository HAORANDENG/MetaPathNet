import argparse
import glob
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm

from root import ROOT
import torch.multiprocessing
import path_sampler

from mag240m import MAG240M

torch.multiprocessing.set_sharing_strategy('file_system')

class MetaPathNet(LightningModule):
    '''
    Simple implementation of PathNet
    '''

    def __init__(self, feature_length, hidden_size, out_size, num_relations, dropout, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(MetaPathNet, self).__init__()
        self.save_hyperparameters()
        self.num_relation = num_relations
        self.feature_length, self.hidden_size, self.out_size \
            = feature_length, hidden_size, out_size
        self.dropout = dropout

        self.fc0 = Linear(feature_length, hidden_size)

        # torch.nn.LSTM ref:  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        self.LSTM = torch.nn.LSTM(hidden_size,
                                  hidden_size,
                                  batch_first=True,
                                  dropout=self.dropout)

        self.mlp = Sequential(
            Linear(2 * hidden_size, hidden_size),
            BatchNorm1d(hidden_size),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_size, out_size),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x_path, x_type):
        batch_size, num_w, walk_len, _ = x_path.shape

        x_path = self.fc0(x_path)  # B, W, L, H
        ego = x_path[:, 0, 0].clone()  # B, H
        x_path = x_path.reshape(batch_size*num_w, walk_len, -1)
        print(f"x_path.shape: {x_path.shape}")
        _, (h_n, _) = self.LSTM(x_path) # h_n.shape:
        print(f"batch_size: {batch_size}")
        print(f"num_w: {num_w}")
        print(f"hidden_size: {self.hidden_size}")
        print(f"h_n.shape: {h_n.shape}")


        return h_n

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x_path, batch.x_type)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x_path, batch.x_type)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x_path, batch.x_type)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    print("[ MetaSage ]")
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='metasage')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.in_memory)

    if not args.evaluate:
        model = MetaPathNet(datamodule.num_features,
                            args.hidden_channels, datamodule.num_classes,
                            datamodule.num_relations, dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                              save_top_k=1)
        trainer = Trainer(gpus=[0], max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}', accelerator='gpu', devices=2)
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=[0], resume_from_checkpoint=ckpt, accelerator='gpu', devices=2)
        model = MetaPathNet.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, f'results/{args.model}',
                                       mode='test-dev')
