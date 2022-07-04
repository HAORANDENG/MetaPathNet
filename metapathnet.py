import argparse
import glob
import os
import os.path as osp
import time
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
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

torch.multiprocessing.set_sharing_strategy('file_system')


class Batch(NamedTuple):
    x_path: Tensor
    x_type: Tensor
    y: Tensor

    def to(self, *args, **kwargs):
        return Batch(
            x_path=self.x_path.to(*args, **kwargs),
            x_type=self.x_type.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
        )


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


class MetaPathSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index, node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, transform: Callable = None, **kwargs):

        edge_index = edge_index.to('cpu')
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']
        # Save for Pytorch Lightning < 1.6:
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        self.adj_t = edge_index
        self.adj_t.storage.rowptr()
        self.transform = transform

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        row, col, val = self.adj_t.coo()
        E = torch.stack([row, col, val], dim=0).numpy()

        path_sampler.init()
        self.ps = path_sampler.MetaPathSampler(E, 1024, 40, 4, 64)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        ret_a, ret_b = self.ps.sample(np.array(batch, dtype=np.int64))
        out = (batch, ret_a, ret_b)
        out = self.transform(*out)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'



class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int,
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))

        if self.in_memory:
            print("flag_1")
            self.x = np.empty((N, self.num_features), dtype=np.float16)
            print("flag_2")
            self.x[:] = x
            print("flag_3")
            self.x = torch.from_numpy(self.x).share_memory_()
            print("flag_4")
        else:
            self.x = x

        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        tmp = self.train_dataloader()

    def train_dataloader(self):
        return MetaPathSampler(self.adj_t,  batch_size=self.batch_size,
                               node_idx=self.train_idx,
                               transform=self.convert_batch,
                               shuffle=True,
                               num_workers=64)

    def val_dataloader(self):
        return MetaPathSampler(self.adj_t, batch_size=self.batch_size,
                               node_idx=self.val_idx,
                               transform=self.convert_batch,
                               num_workers=32)

    def test_dataloader(self):  # Test best validation model once again.
        return MetaPathSampler(self.adj_t, batch_size=self.batch_size,
                               node_idx=self.val_idx,
                               transform=self.convert_batch,
                               num_workers=32)

    def hidden_test_dataloader(self):
        return MetaPathSampler(self.adj_t, batch_size=self.batch_size,
                               node_idx=self.test_idx,
                               transform=self.convert_batch,
                               num_workers=32)

    def convert_batch(self, id, ret_a, ret_b):
        x_path = torch.from_numpy(self.x[ret_a]).to(torch.float)
        x_type = torch.from_numpy(ret_b).to(torch.long)
        y = self.y[id].to(torch.long)
        return Batch(x_path=x_path, x_type=x_type, y=y)

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

        self.fc0 = torch.nn.Linear(feature_length, hidden_size)

        self.dropout = dropout
        self.LSTM = torch.nn.LSTM(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(2 * hidden_size, out_size)
        self.nets = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for i in range(num_relations)])

        self.attw = torch.nn.Linear(2 * hidden_size, 1)
        self.Lrelu = torch.nn.LeakyReLU()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x_path, x_type):
        batch_size, num_w, walk_len, _ = x_path.shape

        x_path = self.fc0(x_path) # B, W, L, H
        ego = x_path[:, 0, 0].clone()

        x_path = x_path.reshape(batch_size * num_w, walk_len, self.hidden_size)
        x_path = torch.flip(x_path, dims=[0]).reshape(batch_size * num_w * walk_len, self.hidden_size)


        x_type = x_type.view(batch_size * num_w * walk_len)

        nei_list = []
        for layer in self.nets:
            nei_l = layer(x_path)
            nei_list.append(nei_l)
        x_path = torch.stack(nei_list, dim=1)

        indxx = torch.arange(batch_size * num_w * walk_len, dtype=torch.long)
        x_path = x_path[indxx, x_type].view(
            batch_size * num_w, walk_len, self.hidden_size).transpose(0, 1)
        x_path = F.dropout(x_path, p=self.dropout, training=self.training)
        x_path, (h_n, c_n) = self.LSTM(x_path)
        h_n = h_n.transpose(0, 1).view(
            num_w, batch_size, -1)  # [V, num_of_walks, H]

        h_n = torch.mean(h_n, dim=0)
        layer1 = torch.cat((ego, h_n), dim=1)  # [V, 2*H]
        layer1 = F.dropout(layer1, p=self.dropout, training=self.training)
        dout = self.fc2(layer1)
        return dout

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
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
