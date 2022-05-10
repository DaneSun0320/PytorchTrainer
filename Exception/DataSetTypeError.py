#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :PytorchDemo_LetNet
# @FileName     :DataSetTypeError
# @CreatTime    :2022/5/10 14:14 
# @CreatUser    :DaneSun

class DataSetTypeError(Exception):
    "This Exception is for DataSetTypeError"
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message