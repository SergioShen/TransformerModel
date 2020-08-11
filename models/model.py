#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 01:03 2020/6/12
# @Author: Sijie Shen
# @File: model
# @Project: TransformerModel

import torch
import torch.nn as nn
import numpy as np
from .transformer import Transformer, GPT
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

        # Build transformer_params model
        self.transformer = Transformer(self.d_model, self.nhead, self.num_encoder_layers, self.num_decoder_layers,
                                       self.dim_feedforward, self.drouput, self.activation)
        self.positional_encoding = PositionalEncoding(self.d_model, self.drouput)

        # Build embedding and out layer
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
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
        if not self.share_vocab:
            self.tgt_embedding.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        if not self.weight_tying:
            self.out.weight.data.uniform_(-init_range, init_range)

    def _get_init_output_ids(self, decode_length, batch_size):
        output_ids = torch.ones(decode_length, batch_size, dtype=torch.int64)
        output_ids[0].fill_(2)

        return output_ids


class GPTModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.vocab_size = model_params['vocab_size']
        self.d_model = model_params['d_model']
        self.nhead = model_params['nhead']
        self.num_encoder_layers = model_params['num_encoder_layers']
        self.dim_feedforward = model_params['dim_feedforward']
        self.drouput = model_params['drouput']
        self.activation = model_params['activation']
        self.weight_tying = model_params['weight_tying']

        # Build transformer_params model
        self.gpt = GPT(self.d_model, self.nhead, self.num_encoder_layers, self.dim_feedforward,
                       self.drouput, self.activation)
        self.positional_encoding = PositionalEncoding(self.d_model, self.drouput)

        # Build embedding and out layer
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embedding.reset_parameters()
        self.out = nn.Linear(self.d_model, self.vocab_size)
        if self.weight_tying:
            self.out.weight = self.embedding.weight

        self._reset_parameters()

    def forward(self, input_ids):
        """
        Model forward
        :param input_ids: (S, N)
        :return: logits: (T, N, E)
        """
        device = input_ids.device

        src_embedded = self.embedding(input_ids)

        src_mask = self.gpt.generate_square_subsequent_mask(input_ids.size(0)).to(device)
        src_key_padding_mask = input_ids.transpose(1, 0) == 1

        output = self.gpt(src_embedded, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.out(output)

        return logits

    def inference(self, input_ids, max_decode_length, beam_size):
        """
        Model inference
        :param input_ids: (S, N)
        :param max_decode_length: Max length of decoder steps
        :param beam_size: beam search size
        :return: output_ids: (T, N); output_lengths: (N)
        """
        device = input_ids.device
        input_length = input_ids.size(0)
        batch_size = input_ids.size(1)
        assert batch_size == 1

        output_seqs = list()
        output_lengths = list()
        output_seq_scores = list()

        output_ids = self._get_init_output_ids(max_decode_length, 1).to(device)  # (P)
        output_scores = torch.tensor([0], dtype=torch.float32).to(device)  # (P)

        src_mask = self.gpt.generate_square_subsequent_mask(max_decode_length + input_length).to(device)

        for i in range(max_decode_length):
            prefix_size = output_ids.size(1)
            expanded_input_ids = input_ids.expand([input_length, prefix_size])  # (S, P)
            step_input = torch.cat([expanded_input_ids, output_ids], dim=0)  # (S + T, P)
            src_key_padding_mask = step_input.t() == 1
            src_embedded = self.embedding(step_input)
            output = self.gpt(src_embedded, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

            # Compute step top k scores
            step_logits = output[i + input_length - 1]  # (P, D)
            step_scores = torch.log_softmax(step_logits, dim=1)  # (P, D)
            step_top_k_scores, step_top_k_indices = step_scores.topk(beam_size, dim=1)  # (P, B), (P, B)

            step_top_k_scores = step_top_k_scores.reshape(-1)  # (P * B)
            step_top_k_indices = step_top_k_indices.reshape(-1)  # (P * B)

            # Compute global top k scores
            output_scores = output_scores.repeat(beam_size, 1).t().reshape(-1) + step_top_k_scores  # (P * B)
            top_k_scores, top_k_indices = output_scores.topk(beam_size, dim=0)  # (B), (B)
            step_output_ids = step_top_k_indices.index_select(dim=0, index=top_k_indices)  # (B)

            # Update output ids
            prefix_indices = top_k_indices / beam_size  # (B)
            output_ids = torch.index_select(output_ids, dim=1, index=prefix_indices)  # (S, B)
            output_ids[i] = step_output_ids
            output_scores = top_k_scores  # (B)

            # Remove finished items
            step_finish = step_output_ids == 4  # (B)
            keep_indices = list()
            for c, finish in enumerate(step_finish):
                finish = finish.item()
                if finish:
                    if len(output_seqs) < beam_size:
                        output_seqs.append(output_ids[:i + 1, c].cpu().tolist())
                        output_lengths.append(i + 1)
                        output_seq_scores.append(output_scores[c].cpu().item())
                else:
                    keep_indices.append(c)

            keep_indices = torch.tensor(keep_indices, dtype=torch.int64).to(device)  # (P)
            output_ids = output_ids.index_select(dim=1, index=keep_indices)  # (S, P)
            output_scores = output_scores.index_select(dim=0, index=keep_indices)  # (P)

            if len(output_seqs) == beam_size:
                break

        for i in range(beam_size - len(output_seqs)):
            output_seqs.append(output_ids[:, i].cpu().tolist())
            output_lengths.append(max_decode_length)
            output_seq_scores.append(output_scores[i].cpu().item())

        return output_ids, output_lengths, output_seq_scores

    def _reset_parameters(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        if not self.weight_tying:
            self.out.weight.data.uniform_(-init_range, init_range)

    def _get_init_output_ids(self, decode_length, batch_size):
        output_ids = torch.ones(decode_length, batch_size, dtype=torch.int64)

        return output_ids
