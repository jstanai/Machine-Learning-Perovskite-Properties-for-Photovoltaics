#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

import numpy as np

def mean_relative_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

