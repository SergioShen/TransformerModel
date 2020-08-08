#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 00:36 2020/6/14
# @Author: Sijie Shen
# @File: gpt_main.py
# @Project: TransformerModel

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import random
import os
import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu

from utils.logger import get_logger
from utils.trainer import Trainer
from utils.metrics import get_correct_num
from models.model import GPTModel


class MyTrainer(Trainer):
    def __init__(self, model, optimizer, lr_scheduler, loss_function, logger, writer, train_params):
        super().__init__(model, optimizer, lr_scheduler, loss_function, logger, writer, train_params)

    def train_batch(self, batch_data):
        input_ids = batch_data.tokens[:-1]
        target_ids = batch_data.tokens[1:]

        logits = self.model(input_ids)

        loss = self.loss_function(logits.reshape([-1, logits.size(2)]), target_ids.reshape([-1]))
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    def evaluate_batch(self, batch_data):
        input_ids = batch_data.tokens[:-1]
        target_ids = batch_data.tokens[1:]

        logits = self.model(input_ids)

        loss = self.loss_function(logits.reshape([-1, logits.size(2)]), target_ids.reshape([-1]))
        correct, total = get_correct_num(logits.argmax(-1).cpu().numpy(), target_ids.cpu().numpy())
        self.cache['correct'] = self.cache.get('correct', 0) + correct
        self.cache['total'] = self.cache.get('total', 0) + total
        rule_correct, rule_total = get_correct_num(logits.argmax(-1).cpu().numpy(), target_ids.cpu().numpy(),
                                                   ref_condition=lambda x: 4 <= x <= 100)
        self.cache['rule_correct'] = self.cache.get('rule_correct', 0) + rule_correct
        self.cache['rule_total'] = self.cache.get('rule_total', 0) + rule_total

        return loss.item()

    def handle_eval_other_infos(self, name):
        accuracy = self.cache['correct'] / self.cache['total']
        self.logger.info('%s accuracy: %.6f' % (name.capitalize(), accuracy))
        self.writer.add_scalar('accuracy/%s_accuracy' % name, accuracy, self.step)
        self.cache['correct'] = 0
        self.cache['total'] = 0

        rule_accuracy = self.cache['rule_correct'] / self.cache['rule_total']
        self.logger.info('%s rule accuracy: %.6f' % (name.capitalize(), rule_accuracy))
        self.writer.add_scalar('accuracy/%s_rule_accuracy' % name, rule_accuracy, self.step)
        self.cache['rule_correct'] = 0
        self.cache['rule_total'] = 0

    def handle_epoch_other_infos(self):
        self.lr_scheduler.step(self.cache['valid_loss'])

    def inference(self, dataset, name, tgt_vocab, max_decode_length=64):
        pass


def main(args):
    # Read params configuration
    params = json.load(open(args.params))
    model_params = params['model_params']
    train_params = params['train_params']
    output_dir = Path(train_params['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir()
    if len(list(output_dir.iterdir())) != 0 and args.train:
        raise FileExistsError('Output dir \'%s\' is not empty' % output_dir)

    # Set up logger and TensorBoard writer
    if args.train:
        logger = get_logger(output_dir / 'train.log')
    elif args.inference:
        logger = get_logger(output_dir / 'inference.log')
    logger.debug('PID: %d', os.getpid())
    logger.info('Using params file: %s' % args.params)
    logger.info(json.dumps(params))
    if args.train:
        writer = SummaryWriter(str(output_dir))
        writer.add_text('Params', json.dumps(params, indent='  '), 0)
    elif args.inference:
        writer = None
    # Set random seed
    seed = 1911
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
    logger.debug('Set random seed to %d', seed)

    # Load vocabulary and dataset
    logger.debug('Loading vocabulary...')
    field = torchtext.data.field.Field(init_token='<sos>', eos_token='<eos>')
    field.vocab = pickle.load(Path(train_params['dataset']['vocab_path']).open('rb'))
    logger.info('Vocab loaded, vocab size: %d' % len(field.vocab))
    assert len(field.vocab) == model_params['vocab_size']
    logger.debug('Loading dataset...')

    datasets = dict()
    if args.train:
        for name in ['train', 'valid', 'test']:
            dataset = torchtext.data.TabularDataset(Path(train_params['dataset'][name]), 'json',
                                                    {'tokens': ('tokens', field)},
                                                    filter_pred=lambda x: len(x.tokens) <= 600)
            datasets[name] = dataset
            logger.debug('%s size: %d' % (name.capitalize(), len(dataset)))
    elif args.inference:
        for name in ['valid', 'test']:
            dataset = torchtext.data.TabularDataset(Path(train_params['dataset'][name]), 'json',
                                                    {'src_action_tokens': ('input_ids', field),
                                                     'tgt_action_tokens': ('output_ids', field)},
                                                    filter_pred=lambda x: len(x.tokens) <= 600)
            datasets[name] = dataset
            logger.debug('%s size: %d' % (name.capitalize(), len(dataset)))

    # Build model
    logger.debug('Building model...')
    model = GPTModel(model_params)
    if train_params['optimizer'] == 'AdamW' and not hasattr(optim, 'AdamW'):
        from utils.adamW import AdamW
        optimizer = AdamW(model.parameters(), **train_params['optimizer_args'])
    else:
        optimizer = getattr(optim, train_params['optimizer'])(model.parameters(), **train_params['optimizer_args'])
    lr_scheduler = getattr(optim.lr_scheduler, train_params['lr_scheduler'])(
        optimizer, **train_params['lr_scheduler_args'])
    loss_function = getattr(nn, train_params['loss_function'])(**train_params['loss_function_args'])
    if torch.cuda.is_available():
        model.cuda()
    logger.debug('Model built')

    # Train model
    trainer = MyTrainer(model, optimizer, lr_scheduler, loss_function, logger, writer, train_params)
    if args.load is not None:
        logger.info('Loading model from %s', args.load)
        trainer.load_model(args.load)
    logger.info('Model loaded')
    if args.train:
        logger.info('Training begins at %d-th epoch...', train_params['start_epoch'])
        trainer.train(datasets['train'], datasets['valid'], datasets['test'])
    if args.inference:
        model.eval()
        logger.info('Inference begins...')
        trainer.inference(datasets['valid'], 'valid', field.vocab)
        trainer.inference(datasets['test'], 'test', field.vocab)


if __name__ == '__main__':
    parser = ArgumentParser('Tree positional encoding experiment main function.')
    parser.add_argument('-p', '--params', action='store',
                        help='Path of configuration file, should be a .json file')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train a model')
    parser.add_argument('-i', '--inference', action='store_true',
                        help='Inference the model')
    parser.add_argument('-l', '--load', action='store', default=None,
                        help='Load a model from given path')
    args = parser.parse_args()

    # Check arguments
    if args.train + args.inference != 1:
        print('Train and inference can\'t be set as True or False simultaneously')
        exit(-1)
    if args.inference and not args.load:
        print('Should specify a checkpoint for inference')
        exit(-1)

    main(args)
