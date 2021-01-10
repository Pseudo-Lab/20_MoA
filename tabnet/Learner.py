import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from torch.cuda.amp import GradScaler, autocast
from contextlib import ExitStack

from scripts.utils import init_logger, init_tb_logger
from scripts.losses import LabelSmoothing, SmoothBCEwLogits
from scripts.metric import AverageMeter


class Learner:
    def __init__(self, model, train_loader, valid_loader, fold, config, seed):
        self.config = config
        self.seed = seed
        self.device = self.config.device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(self.device)

        self.fold = fold
        self.logger = init_logger(config.log_dir, f'train_seed{self.seed}_fold{self.fold}.log')
        self.tb_logger = init_tb_logger(config.log_dir, f'train_seed{self.seed}_fold{self.fold}')
        if self.fold == 0:
            self.log('\n'.join([f"{k} = {v}" for k, v in self.config.__dict__.items()]))

        self.criterion = SmoothBCEwLogits(smoothing=self.config.smoothing)
        self.evaluator = nn.BCEWithLogitsLoss()
        self.summary_loss = AverageMeter()
        self.history = {'train': [], 'valid': []}

        self.optimizer = Adam(self.model.parameters(), lr=config.lr,
                              weight_decay=self.config.weight_decay)
        self.scheduler = OneCycleLR(optimizer=self.optimizer, pct_start=0.1, div_factor=1e3,
                                    max_lr=1e-2, epochs=config.n_epochs, steps_per_epoch=len(train_loader))
        self.scaler = GradScaler() if config.fp16 else None

        self.epoch = 0
        self.best_epoch = 0
        self.best_loss = np.inf

    def train_one_epoch(self):
        self.model.train()
        self.summary_loss.reset()
        iters = len(self.train_loader)
        for step, (g_x, c_x, cate_x, labels, non_labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # self.tb_logger.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'],
            #                           iters * self.epoch + step)
            labels = labels.to(self.device)
            non_labels = non_labels.to(self.device)
            g_x = g_x.to(self.device)
            c_x = c_x.to(self.device)
            cate_x = cate_x.to(self.device)
            batch_size = labels.shape[0]

            with ExitStack() as stack:
                if self.config.fp16:
                    auto = stack.enter_context(autocast())
                outputs = self.model(g_x, c_x, cate_x)
                loss = self.criterion(outputs, labels)

            if self.config.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.summary_loss.update(loss.item(), batch_size)
            if self.scheduler.__class__.__name__ != 'ReduceLROnPlateau':
                self.scheduler.step()

        self.history['train'].append(self.summary_loss.avg)
        return self.summary_loss.avg

    def validation(self):
        self.model.eval()
        self.summary_loss.reset()
        iters = len(self.valid_loader)
        for step, (g_x, c_x, cate_x, labels, non_labels) in enumerate(self.valid_loader):
            with torch.no_grad():
                labels = labels.to(self.device)
                g_x = g_x.to(self.device)
                c_x = c_x.to(self.device)
                cate_x = cate_x.to(self.device)
                batch_size = labels.shape[0]
                outputs = self.model(g_x, c_x, cate_x)
                loss = self.evaluator(outputs, labels)

                self.summary_loss.update(loss.detach().item(), batch_size)

        self.history['valid'].append(self.summary_loss.avg)
        return self.summary_loss.avg

    def fit(self, epochs):
        self.log(f'Start training....')
        for e in range(epochs):
            t = time.time()
            loss = self.train_one_epoch()

            # self.log(f'[Train] \t Epoch: {self.epoch}, loss: {loss:.6f}, time: {(time.time() - t):.2f}')
            self.tb_logger.add_scalar('Train/Loss', loss, self.epoch)

            t = time.time()
            loss = self.validation()

            # self.log(f'[Valid] \t Epoch: {self.epoch}, loss: {loss:.6f}, time: {(time.time() - t):.2f}')
            self.tb_logger.add_scalar('Valid/Loss', loss, self.epoch)
            self.post_processing(loss)

            self.epoch += 1
        self.log(f'best epoch: {self.best_epoch}, best loss: {self.best_loss}')
        return self.history

    def post_processing(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = self.epoch

            self.model.eval()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_loss': self.best_loss,
                'epoch': self.epoch,
            }, f'{os.path.join(self.config.log_dir, f"{self.config.name}_seed{self.seed}_fold{self.fold}.pth")}')
            self.log(f'best model: {self.epoch} epoch - loss: {loss:.6f}')

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, text):
        self.logger.info(text)
