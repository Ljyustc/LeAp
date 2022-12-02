# -*- coding:utf-8 -*-

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Hierarchical Math Solver')

    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    parser.add_argument('--checkpoint', type=str, dest='checkpoint', default=None)
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    parser.add_argument('--log', type=str, dest='log', default=None)
    parser.add_argument('--test-log', type=str, dest='test_log', default=None)
    parser.add_argument('--seed', type=int, dest='seed', default=10)
    parser.add_argument('--run-flag', type=str, dest='run_flag',default='train')

    parser.add_argument('--epoch', type=int, dest='epoch', default=80)
    parser.add_argument('--batch', type=int, dest='batch', default=64)
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    parser.add_argument('--weight-decay', type=float, dest='weight_decay', default=1e-5)
    parser.add_argument('--step', type=int, dest='step', default=20)
    parser.add_argument('--gamma', type=float, dest='gamma', default=0.5)
    parser.add_argument('--beam', type=int, dest='beam', default=1)

    parser.add_argument('--embed', type=int, dest='embed', default=128)
    parser.add_argument('--hidden', type=int, dest='hidden', default=512)
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.5)
    parser.add_argument('--fold', type=int, dest='fold', default=0)
    parser.add_argument('--temp', type=float, dest='temp', default=0.1)
    parser.add_argument('--prior_prob', type=float, dest='prior_prob', default=0.1)
    parser.add_argument('--thre', type=float, dest='thre', default=0.5)
    parser.add_argument('--kr', type=float, dest='kr', default=0.2)

    args = parser.parse_args()

    args.use_cuda = args.cuda_id is not None
    return args
