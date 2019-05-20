# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 21:27
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'kitti':
            return '/data/wangxin/KITTI'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
