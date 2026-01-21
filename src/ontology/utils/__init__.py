#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import joblib
from .data_utils import *
from .ontology import *
from .label_list import *
from .sample import get_sample
from .evaluation import *

def sigmoid(X):
    return 1.0/(1.0 + np.exp(-X))


def save(save_path, content, key='model'):
    joblib.dump(content, save_path)


def load(load_path, key='model'):
    return joblib.load(load_path)
