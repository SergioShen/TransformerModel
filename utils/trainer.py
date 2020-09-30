#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 23:34 2020/6/13
# @Author: Sijie Shen
# @File: trainer.py
# @Project: TransformerModel

import torch
import torchtext
from pathlib import Path
from abc import abstractmethod


class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_function, logger, writer, train_params):
        self.model = model
        self.logger = logger
        self.writer = writer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.train_params = train_params
        self.output_dir = Path(train_params['output_dir'])
        self.batch_size = train_params['batch_size']
        self.n_epochs = train_params['n_epochs']
        self.start_epoch = train_params['start_epoch']
        self.grad_clip = train_params['grad_clip']
        self.print_step = train_params['print_step']
        self.cache = dict()
        self.step = 0

    def save_model(self, epoch):
        checkpoint = dict()
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        checkpoint['loss_function'] = self.loss_function

        checkpoint['step'] = self.step
        checkpoint['epoch'] = epoch
        checkpoint['batch_size'] = self.batch_size

        save_path = self.output_dir / ('%0.3d.epoch.pt' % epoch)
        torch.save(checkpoint, save_path)

    def load_model(self, model_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location='cuda:0')
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        if self.optimizer and checkpoint['optimizer']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.lr_scheduler and checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if self.loss_function and checkpoint['loss_function']:
            self.loss_function = checkpoint['loss_function']

        self.step = checkpoint['step']
        self.start_epoch = checkpoint['epoch'] + 1
        self.batch_size = checkpoint['batch_size']

    def train(self, train_data, valid_data=None, test_data=None, inference_every_epoch=False):
        self.model.train()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        steps_per_epoch = len(train_data) // self.batch_size
        total_steps = steps_per_epoch * self.n_epochs
        self.logger.info('Steps per epoch: %d, total steps: %d' % (steps_per_epoch, total_steps))

        self.step = (self.start_epoch - 1) * steps_per_epoch + 1
        print_loss = 0
        epoch_loss = 0
        iterator = torchtext.data.iterator.Iterator(train_data, self.batch_size, shuffle=True, device=device)

        for _ in range(self.start_epoch - 1):
            iterator.init_epoch()
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            self.logger.info('Epoch: %d, step: %d' % (epoch, self.step))
            iterator.init_epoch()
            for batch_data in iter(iterator):
                step_loss = self.train_batch(batch_data)
                print_loss += step_loss
                epoch_loss += step_loss
                self.writer.add_scalar('loss/step_loss', step_loss, self.step)

                # Print current information every `print_every` steps
                if self.step % self.print_step == 0:
                    print_loss_avg = print_loss / self.print_step
                    print_loss = 0
                    self.logger.info('Progress %.2f%%, Train loss: %.6f', self.step / total_steps * 100, print_loss_avg)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.step)
                    self.handle_print_other_infos()

                self.step += 1

            # Epoch finish
            # Checkpoint
            self.save_model(epoch)
            self.logger.info('Epoch: %d, checkpoint saved', epoch)

            epoch_loss_avg = epoch_loss / steps_per_epoch
            epoch_loss = 0
            self.cache['epoch_loss_avg'] = epoch_loss_avg
            self.writer.add_scalar('loss/train_loss', epoch_loss_avg, self.step)
            self.logger.info('Finished epoch %d: Train loss: %.6f' % (epoch, epoch_loss_avg))

            # Evaluate the model with dev and test data
            if valid_data:
                loss = self.evaluate(valid_data)
                self.cache['valid_loss'] = loss
                self.writer.add_scalar('loss/valid_loss', loss, self.step)
                self.logger.info('Valid loss: %.6f' % loss)
                self.handle_eval_other_infos('valid')
                self.model.train()

            if test_data:
                loss = self.evaluate(test_data)
                self.cache['test_loss'] = loss
                self.writer.add_scalar('loss/test_loss', loss, self.step)
                self.logger.info('Test loss: %.6f' % loss)
                self.handle_eval_other_infos('test')
                self.model.train()

            # Inference
            if inference_every_epoch:
                if valid_data:
                    self.inference(valid_data)
                if test_data:
                    self.inference(test_data)

            self.handle_epoch_other_infos()

        self.writer.export_scalars_to_json(self.output_dir / 'all_scalars.json')
        self.writer.close()

    def evaluate(self, dataset, eval_batch=None):
        self.model.eval()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        iterator = torchtext.data.iterator.Iterator(dataset, self.batch_size, device=device)
        total_loss = 0
        batch_elapsed = 0

        with torch.no_grad():
            for batch_data in iter(iterator):
                if eval_batch is not None and batch_elapsed >= eval_batch:
                    break
                step_loss = self.evaluate_batch(batch_data)
                batch_elapsed += 1
                total_loss += step_loss

        total_loss /= batch_elapsed

        return total_loss

    @abstractmethod
    def train_batch(self, batch_data):
        """
        Train a batch, update the model parameters and return loss
        In the meantime, some intermediate data can be stored in self.cache for later print
        :param batch_data: a batch of data from DataLoader
        :return: loss
        """
        pass

    @abstractmethod
    def evaluate_batch(self, batch_data):
        """
        Evaluate a batch and return loss
        In the meantime, some intermediate data can be stored in self.cache for later print
        :param batch_data: a batch of data from DataLoader
        :return: loss
        """
        pass

    @abstractmethod
    def inference(self, *args):
        """
        Inference a dataset
        :return:
        """
        pass

    def handle_print_other_infos(self):
        """
        Print other infos other than loss value in every `print_step`
        The function can use intermediate data stored in self.cache by train_batch and evaluate_batch
        :return: None
        """
        pass

    def handle_eval_other_infos(self, name):
        """
        Print other infos other than loss value in the end of evaluating a dataset
        The function can use intermediate data stored in self.cache by train_batch and evaluate_batch
        :return None
        """
        pass

    def handle_epoch_other_infos(self):
        """
        Print other infos other than loss value in the end of every epoch
        The function can use intermediate data stored in self.cache by train_batch and evaluate_batch
        :return None
        """
        pass
