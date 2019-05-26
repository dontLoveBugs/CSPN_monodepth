#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-19 15:54
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""
from functools import cmp_to_key

x = [[1,2], [2, 1], [3, 4]]

def mycmp(x, y):
    if x[1] == y[1]:
        return x[0] - y[0]
    return x[1] - y[1]

print(x)
x= sorted(x, key=cmp_to_key(mycmp))
print(x)

y = set(0)


from queue import PriorityQueue