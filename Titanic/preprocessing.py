# coding:utf-8
'''
Created on 2018/1/30.

@author: chk01
'''

import pandas as pd
from pathlib import Path


class DataClean(object):
    def __init__(self, file):
        '''
        :param file:csv file
        '''
        self.X = pd.read_csv(file)
