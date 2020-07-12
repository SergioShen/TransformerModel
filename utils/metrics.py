#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 02:42 2020/6/13
# @Author: Sijie Shen
# @File: metrics.py
# @Project: TransformerModel


def get_correct_num(hyp, ref, pad_id=1, unknown_true=False, ref_condition=None):
    valid = ref != pad_id
    if ref_condition:
        valid = valid & (ref_condition(ref))
    if unknown_true:
        correct = (hyp == ref) & valid
    else:
        correct = (hyp == ref) & valid & (ref != 0)

    return correct.sum(), valid.sum()
