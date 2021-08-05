# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:36:25 2021

@author: 成世杰
"""

import argparse

def get_parser():
    #1
   # parser = argparse.ArgumentParser(description='learning argparse')
   # parser.add_argument('--name',default='Great')
    #2
   # parser=argparse.ArgumentParser(description='calculate square of a given number')
   #parser.add_argument('--number',type=int)
    #3
    parser = argparse.ArgumentParser(description='learning argparse')
    parser.add_argument('--arch',required=True,choices=['alexnet','vgg'])
    return parser


if __name__ == '__main__':
    parser=get_parser()
    args=parser.parse_args()
   # name = args.name
   # print('hello{}'.format(name))
   # res=args.number**2
   # print('squre of {}is{}'.format(args.number,res))
    print('the arch of cnn is{}'.format(args.arch))
