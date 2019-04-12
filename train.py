#!/usr/bin/env python
"""Train models."""
import onmt.opts as opts
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    if opt.gpu>-1:  # case GPU
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
