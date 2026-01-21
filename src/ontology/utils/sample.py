#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2017/1/16
@author yrh

"""

import sys
import random
import configparser
sys.path.append("/home/liuhc/GOCurator-main")
from utils import get_type_list, NS_ID


__all__ = ['get_sample']


def get_sample(sample_file, times=10000):
    with open(sample_file) as fp:
        samples = [line.strip() for line in fp]
    length = len(samples) // times
    return [samples[i*length: (i+1)*length] for i in range(times)]


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    times = conf.getint('t-test', 'times', fallback=10000)
    for ns in NS_ID:
        type_list = list(get_type_list(conf.get('type_file', ns)))
        with open(conf.get('t-test', ns), 'w') as fp:
            for _ in range(len(type_list) * times):
                print(random.choice(type_list), file=fp)


if __name__ == '__main__':
    main(sys.argv[1:])
