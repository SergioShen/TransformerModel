#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 00:36 2020/6/14
# @Author: Sijie Shen
# @File: main.py
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
from models.model import TransformerModel


class MyTrainer(Trainer):
    def __init__(self, model, optimizer, lr_scheduler, loss_function, logger, writer, train_params):
        super().__init__(model, optimizer, lr_scheduler, loss_function, logger, writer, train_params)

    def train_batch(self, batch_data):
        input_ids = batch_data.input_ids
        output_ids = batch_data.output_ids[:-1]
        target_ids = batch_data.output_ids[1:]

        logits = self.model(input_ids, output_ids)

        loss = self.loss_function(logits.reshape([-1, logits.size(2)]), target_ids.reshape([-1]))
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    def evaluate_batch(self, batch_data):
        input_ids = batch_data.input_ids
        output_ids = batch_data.output_ids[:-1]
        target_ids = batch_data.output_ids[1:]

        logits = self.model(input_ids, output_ids)

        loss = self.loss_function(logits.reshape([-1, logits.size(2)]), target_ids.reshape([-1]))
        correct, total = get_correct_num(logits.argmax(-1).cpu().numpy(), target_ids.cpu().numpy())
        self.cache['correct'] = self.cache.get('correct', 0) + correct
        self.cache['total'] = self.cache.get('total', 0) + total

        return loss.item()

    def handle_eval_other_infos(self, name):
        accuracy = self.cache['correct'] / self.cache['total']
        self.logger.info('%s accuracy: %.6f' % (name.capitalize(), accuracy))
        self.writer.add_scalar('accuracy/%s_accuracy' % name, accuracy, self.step)
        self.cache['correct'] = 0
        self.cache['total'] = 0

    def handle_epoch_other_infos(self):
        self.lr_scheduler.step(self.cache['valid_loss'])

    def inference(self, dataset, name, tgt_vocab, max_decode_length=64):
        self.model.eval()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        iterator = torchtext.data.iterator.Iterator(dataset, self.batch_size, device=device)

        hypotheses = list()
        references = list()

        for data_piece in dataset:
            references.append([data_piece.output_ids])

        with torch.no_grad():
            for batch_data in iter(iterator):
                input_ids = batch_data.input_ids
                output_ids, output_lengths = self.model.inference(input_ids, max_decode_length)
                output_ids = output_ids.transpose(1, 0).cpu().numpy()
                for i in range(len(output_ids)):
                    remove_padding = output_ids[i][:output_lengths[i]].tolist()
                    if remove_padding[0] == 2:
                        remove_padding = remove_padding[1:]
                    if remove_padding[-1] == 3:
                        remove_padding = remove_padding[:-1]
                    hypo = [tgt_vocab.itos[idx] for idx in remove_padding]
                    hypotheses.append(hypo)
        bleu_score = corpus_bleu(references, hypotheses)
        self.logger.info('%s BLEU: %.6f' % (name.capitalize(), bleu_score))


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
    src_field = torchtext.data.field.Field(init_token='<sos>', eos_token='<eos>')
    src_field.vocab = pickle.load(Path(train_params['dataset']['src_vocab_path']).open('rb'))
    assert len(src_field.vocab) == model_params['src_vocab_size']
    tgt_field = torchtext.data.field.Field(init_token='<sos>', eos_token='<eos>', is_target=True)
    tgt_field.vocab = pickle.load(Path(train_params['dataset']['tgt_vocab_path']).open('rb'))
    assert len(tgt_field.vocab) == model_params['tgt_vocab_size']
    logger.info('Vocab loaded, src vocab size: %d, tgt vocab size: %d' % (len(src_field.vocab), len(tgt_field.vocab)))
    logger.debug('Loading dataset...')

    datasets = dict()
    if args.train:
        dataset_names = ['train', 'valid', 'test']
    elif args.inference:
        dataset_names = ['valid', 'test']
    for name in dataset_names:
        dataset = torchtext.data.TabularDataset(Path(train_params['dataset'][name]), 'json',
                                                {'src_tokens': ('input_ids', src_field),
                                                 'tgt_tokens': ('output_ids', tgt_field)},
                                                filter_pred=lambda x: len(x.input_ids) + len(x.output_ids) <= 800)
        datasets[name] = dataset
        logger.debug('%s size: %d' % (name.capitalize(), len(dataset)))

    # Build model
    logger.debug('Building model...')
    model = TransformerModel(model_params)
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
        trainer.inference(datasets['valid'], 'valid', tgt_field.vocab)
        trainer.inference(datasets['test'], 'test', tgt_field.vocab)


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
