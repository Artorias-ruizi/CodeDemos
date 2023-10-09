# -*- coding: utf-8 -*-
# @File  : argExample.py
# @Author: Ruizi Wu
# @Time  : 2022/7/5 18:17
# @Desc  :

import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')

    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')

    parser.set_defaults(
        input_dim=10,
        hidden_dim=20,
        output_dim=20,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        method='base',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print(args)
