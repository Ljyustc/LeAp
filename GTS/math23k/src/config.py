# -*- coding:utf-8 -*-

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='RKLF-Graph2Tree')
    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    parser.add_argument('--temp', type=float, dest='temp', default=0.1)
    parser.add_argument('--prior_prob', type=float, dest='prior_prob', default=0.1)
    parser.add_argument('--thre', type=float, dest='thre', default=0.5)
    parser.add_argument('--kr', type=float, dest='kr', default=0.2)
    args = parser.parse_args()
    return args

args = get_args()