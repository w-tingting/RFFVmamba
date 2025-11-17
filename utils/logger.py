# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, name=''):
    """Create a logger for single-process training.

    The ``dist_rank`` argument is kept for backwards compatibility but is not
    used to branch behaviour, so the same logger works whether or not a
    distributed setup is active.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name or __name__)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = (
        colored('[%(asctime)s %(name)s]', 'green') +
        colored('(%(filename)s %(lineno)d)', 'yellow') +
        ': %(levelname)s %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
