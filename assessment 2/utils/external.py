# Copyright (C), Weights & Biases
# Please DO NOT share.

import pickle


def unpickle(file_name):
    """unpickle function from CIFAR10 webpage"""
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#
# external.py ends here
