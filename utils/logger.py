#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 00:57 2020/6/14
# @Author: Sijie Shen
# @File: logger
# @Project: TransformerModel

import logging
from datetime import datetime, timezone, timedelta


def beijing_time_converter(*args):
    utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_time = utc_time.astimezone(timezone(timedelta(hours=8)))
    return beijing_time.timetuple()


def get_logger(log_path=None, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    formatter.converter = beijing_time_converter

    # File logger
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode=mode)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
