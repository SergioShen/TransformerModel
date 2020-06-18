#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 01:03 2020/6/12
# @Author: Sijie Shen
# @File: model
# @Project: TransformerModel

import torch
import torch.nn as nn
import numpy as np
from .transformer import Transformer
from .module import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.src_vocab_size = model_params['src_vocab_size']
        self.tgt_vocab_size = model_params['tgt_vocab_size']
        self.d_model = model_params['d_model']
        self.nhead = model_params['nhead']
        self.num_encoder_layers = model_params['num_encoder_layers']
        self.num_decoder_layers = model_params['num_decoder_layers']
        self.dim_feedforward = model_params['dim_feedforward']
        self.drouput = model_params['drouput']
        self.activation = model_params['activation']
        self.share_vocab = model_params['share_vocab']
        self.weight_tying = model_params['weight_tying']

        # Build transformer model
        self.transformer = Transformer(self.d_model, self.nhead, self.num_encoder_layers, self.num_decoder_layers,
                                       self.dim_feedforward, self.drouput, self.activation)
        self.positional_encoding = PositionalEncoding(self.d_model, self.drouput)

        # Build embedding and out layer
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        self.src_embedding.reset_parameters()
        if self.share_vocab:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.d_model)
        self.out = nn.Linear(self.d_model, self.tgt_vocab_size)
        if self.weight_tying:
            self.out.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def forward(self, input_ids, output_ids):
        """
        Model forward
        :param input_ids: (S, N)
        :param output_ids: (T, N)
        :return: logits: (T, N, E)
        """
        device = input_ids.device

        src_embedded = self.src_embedding(input_ids)
        tgt_embedded = self.tgt_embedding(output_ids)

        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(output_ids.size(0)).to(device)
        memory_mask = None

        src_key_padding_mask = input_ids.transpose(1, 0) == 1
        tgt_key_padding_mask = output_ids.transpose(1, 0) == 1
        memory_key_padding_mask = src_key_padding_mask

        output = self.transformer(src_embedded, tgt_embedded, src_mask=src_mask, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        logits = self.out(output)

        return logits

    def inference(self, input_ids, max_decode_length):
        """
        Model inference
        :param input_ids: (S, N)
        :param max_decode_length: Max length of decoder steps
        :return: output_ids: (T, N)
        """
        device = input_ids.device
        batch_size = input_ids.size(1)

        src_embedded = self.src_embedding(input_ids)
        output_ids = self._get_init_output_ids(max_decode_length, batch_size).to(device)
        decode_finish = np.asarray([False] * batch_size, dtype=np.bool)
        output_lengths = np.asarray([max_decode_length] * batch_size, dtype=np.int32)

        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(max_decode_length).to(device)
        memory_mask = None

        src_key_padding_mask = input_ids.transpose(1, 0) == 1
        tgt_key_padding_mask = output_ids.transpose(1, 0) == 1
        memory_key_padding_mask = src_key_padding_mask

        memory = self.transformer.encoder(src_embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        for i in range(max_decode_length - 1):
            tgt_embedded = self.tgt_embedding(output_ids)
            output = self.transformer.decoder(tgt_embedded, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)
            step_logits = output[i]  # (N, E)
            step_pred = step_logits.argmax(1)
            output_ids[i + 1] = step_pred

            # Check finish
            finish = (step_pred == 4).cpu().numpy()
            output_lengths[finish & (~decode_finish)] = (i + 2)
            decode_finish = decode_finish | finish
            if sum(decode_finish) == batch_size:
                break

        return output_ids, output_lengths

    def _reset_parameters(self):
        init_range = 0.1
        self.src_embedding.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        if not self.weight_tying:
            self.out.weight.data.uniform_(-init_range, init_range)

    def _get_init_output_ids(self, decode_length, batch_size):
        output_ids = torch.ones(decode_length, batch_size, dtype=torch.int64)
        output_ids[0].fill_(2)

        return output_ids
