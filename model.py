from models.net import Net
from utils.dataset import BlogDataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn as nn
import random
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

class Model(pl.LightningModule):
    """Loads or creates a classification model, that as an input takes vectorized text embeddings and predicts classes.

    Args:
        X_train: numpy array with embedding vectors from train sample
        y_train: numpy array of labels for train sample
        X_val: numpy array with embedding vectors from validation sample
        y_val: numpy array of labels for validation sample
        hparams: python dictionary with hyperparameters if specified:
                :weights: a sequence of weights for WeightedRandomSampler. (default None)
                :output_path: path for checkpoints to save. (default: './checkpoints/model-outputs')
                :dropout: dropout rate from 0 to 1. (default: 0.0)
                :loss_func: loss function. (default: nn.CrossEntropyLoss())
                :hidden_dim: hidden layer's dimension. (default: 128)
                :batch_size: size of the batch. (default: 32)
                :shuffle: set to True to have the data reshuffled at every epoch. (default: True)
                :num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 4)
                :pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them. (default False)
                :random_state: random seed. (default: 17)
                :device: specifies device. (default: 'cpu') 
                :max_epochs: maxinmum number of epochs. (default: 10)
                :verbose: if True, prints the test results (default: True)
                :monitor: quantity to be monitored by checkpoint_callback. (default: 'avg_val_loss')
                :prefix: prefix for checkpoints. (default: '')
                :early_stop_callback: specifies callback for monitoring a metric and stop training when it stops improving. (default: EarlyStopping)
                :early_stop_monitor: quantity to be monitored by EarlyStopping. (default: 'avg_val_loss')
                :early_stop_min_delta: minimum change in the monitored quantity to qualify as an improvement. (default: 1e-3)
                :early_stop_patience: number of validation epochs with no improvement after which training will be stopped. (default: 3)
                :gradient_clip_val: gradient clipping. 0 means don’t clip. (default: 1)
                :gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node. (default: 1)
                :overfit_batches: overfit a percent of training data (float) or a set number of batches (int). (default: 0.0)
                :fast_dev_run: runs 1 batch of train, test and val to find any bugs. (default: False)
                :logger: logger (or iterable collection of loggers) for experiment tracking. (default: False)

    """
    def __init__(self, X_train, y_train, X_val, y_val, hparams = {}):
        super(Model, self).__init__()
        self.hparams = hparams
        self.train_dataset = self.__build_dataset(X_train, y_train)
        self.val_dataset = self.__build_dataset(X_val, y_val)
        self.output_dim = len(np.unique(y_train))
        self.net = self.__build_model()
        self.weights = self.hparams.get('weights', None)
        self.output_path = Path(
            self.hparams.get('output_path', './checkpoints/model-outputs')
        )
        self.trainer_params = self.__get_trainer_params()

    def __build_dataset(self, X, y):
        return BlogDataset(X, y)

    def __build_model(self):
        dropout = self.hparams.get('dropout', 0.0)
        loss_func = self.hparams.get('loss_func', nn.CrossEntropyLoss())
        hidden_dim = self.hparams.get('hidden_dim', 128)
        return Net(768, self.output_dim, loss_func, hidden_dim, dropout)
    
    def forward(self, vector, label=None):
        return self.net(vector, label)

    def train_dataloader(self):
        if self.weights is None:
            loader = DataLoader(self.train_dataset,
                batch_size = self.hparams.get('batch_size', 32),
                shuffle = self.hparams.get('shuffle', True),
                num_workers = self.hparams.get('num_workers', 4),
                pin_memory= self.hparams.get('pin_memory', False)
            )
        else:
            weighted_sampler = WeightedRandomSampler(
                        weights=self.weights,
                        num_samples=len(self.weights),
                        replacement=True
            )
            loader = DataLoader(self.train_dataset,
                batch_size = self.hparams.get('batch_size', 32),
                sampler = weighted_sampler,
                num_workers = self.hparams.get('num_workers', 4),
                pin_memory= self.hparams.get('pin_memory', False)
            )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
            batch_size = self.hparams.get('batch_size', 32),
            shuffle = False,
            num_workers = self.hparams.get('num_workers', 4),
            pin_memory= self.hparams.get('pin_memory', False)
        )
        return loader

    def __set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def fit(self):
        self.__set_seed(self.hparams.get('random_state', 17))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)
    
    def training_step(self, batch, batch_idx):
        device = self.hparams.get('device', 'cpu')
        _, loss = self.forward(
            batch['X'].to(device),
            batch['y'].to(device)
        )
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        device = self.hparams.get('device', 'cpu')
        _, loss = self.eval().forward(
            batch['X'].to(device),
            batch['y'].to(device)
        )
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = [optim.Adam(self.net.parameters(), 
                    lr=self.hparams.get('lr', 1e-3))]
        scheduler = [optim.lr_scheduler.CosineAnnealingLR(optimizer[0], 
                                               T_max=self.hparams.get('max_epochs', 10))]
        return optimizer, scheduler
    
    def __get_trainer_params(self):
        checkpoint_callback = ModelCheckpoint(
                            filepath=self.output_path,
                            verbose=self.hparams.get('verbose', True),
                            monitor=self.hparams.get('monitor', 'avg_val_loss'),
                            mode='min',
                            prefix=self.hparams.get('prefix', ''),
                        )
        early_stop_callback = EarlyStopping(
                            monitor=self.hparams.get(
                                'early_stop_monitor', 'avg_val_loss'),
                            min_delta=self.hparams.get(
                                'early_stop_min_delta', 1e-3),
                            patience=self.hparams.get(
                                'early_stop_patience', 3),
                            verbose=self.hparams.get('verbose', True)
                        )

        trainer_params = {
            'checkpoint_callback': checkpoint_callback,
            'callbacks': [self.hparams.get(
                'early_stop_callback', early_stop_callback)],
            'gradient_clip_val': self.hparams.get(
                "gradient_clip_val", 1),
            'gpus': self.hparams.get('gpus', 1),
            'overfit_batches': self.hparams.get(
                'overfit_batches', 0.0),
            'max_epochs': self.hparams.get(
                'max_epochs', 10),
            'default_root_dir': self.output_path,
            'fast_dev_run': self.hparams.get(
                'fast_dev_run', False),
            'logger': self.hparams.get(
                'logger', False)
        }
        return trainer_params

    @torch.no_grad()
    def predict_eval(self, X_test, y_test):
        device = self.hparams.get('device', 'cpu')
        test_dataset = self.__build_dataset(X_test, y_test)
        loader = DataLoader(test_dataset,
            batch_size = self.hparams.get('batch_size', 32),
            shuffle = False,
            num_workers = self.hparams.get('num_workers ', 4)
        )
        predictions = []
        for batch in tqdm(loader, total = len(loader)):
            preds, _ = self.net.eval()(
                batch['X'].to(device)
            )

            if device=='cuda':
                preds = preds.cpu().detach().numpy()

            predictions += list(preds.argmax(1))
        
        predictions = np.array(predictions)
        
        print(metrics.classification_report(y_test, predictions, digits=3))
        return pd.DataFrame({'y_true': y_test, 'predictions': predictions})
