#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gaohaoyuan
# @Date:   2014-12-23 09:57:12
# @Last Modified by:   gaohaoyuan
# @Last Modified time: 2014-12-23 18:07:57

import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import numpy as np
from matplotlib.pyplot import savefig
import logging
from optparse import OptionParser

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('GEN Batch')
logger.setLevel(logging.INFO)


def parsing_line(line):
    dic = {}
    line = line.strip()
    line = line.split(" ")
    if len(line) == 8:
        if line[5] == 'save' and line[6] == 'dir':
            dic['pass_info'] = int(line[-1].split('pass-')[-1].split('-')[0])
    i = 0
    while i < len(line):
        line_det = line[i].split('=')

        # for training log part
        # logger.debug("line_det %s" % line_det[0])
        if line_det[0] == "Batch":
            dic['test'] = False
            dic['number'] = line[i].split('=')[0]

        if line_det[0] == "AvgCost":
            dic['AvgCost'] = line_det[1]

        if line_det[0] == "CurrentCost":
            logger.debug("Test : %s", line_det[1])
            dic['CurrentCost'] = line_det[1]

        if line_det[0] == 'Eval:':
            i += 1
            dic['Eval'] = line[i].split('=')[1]

        if line_det[0] == "CurrentEval:":
            i += 1
            dic['CurrentEval'] = line[i].split('=')[1]

        # for testing log part
        if line_det[0] == "Test":
            dic['test'] = True
            i += 1
            dic['number'] = line[i].split('=')[1]

        if line_det[0] == "cost":
            dic['cost'] = line_det[1]

        # end
        i += 1

    return dic


def show_image(train_log, test_log, fig_file, fg_name):
    train_x = xrange(len(train_log))
    test_x = xrange(len(test_log))

    logger.info("train dim : %d, test dim: %d " % (len(train_log), len(test_log)))
    pl.figure(1)
    if len(train_log) >= len(test_log):
        x = range(0, len(train_log))
        step = int(len(train_log) / len(test_log))
        x_c = range(0, len(test_x) * step, step)
    if len(train_log) < len(test_log):
        x_c = range(0, len(test_x))
        step = int(len(test_log) / len(train_log))
        x = range(0, len(train_log) * step, step)
    # x = x[0:len(train_log)]
    train_error = []
    test_error = []
    for line in train_log:
        train_error.append(line['Eval'])
    for line in test_log:
        test_error.append(line['Eval'])

    logger.info("X dim %d, Y dim %d" % (len(x), len(train_error)))
    pl.plot(x, train_error, 'k-', label='Training set')
    pl.plot(x_c, test_error, 'r-', label='Testing set')
    pl.xlabel("Epoch")
    pl.ylabel("error classification")
    pl.title(fg_name + " Evaluation Error")
    pl.legend(loc='upper right')
    savefig(fig_file + "_error.jpg")

    pl.figure(2)
    train_cost = []
    test_cost = []
    for line in train_log:
        train_cost.append(line['AvgCost'])
    for line in test_log:
        test_cost.append(line['cost'])
    pl.plot(x, train_cost, 'k-', label='Training set')
    pl.plot(x_c, test_cost, 'r-', label='Testing set')
    ax = plt.gca()
    pl.xlabel("Epoch")
    pl.ylabel("Cost")
    pl.title(fg_name + " Cost")
    pl.legend(loc='upper right')
    savefig(fig_file + "_cost.jpg")


def main(input_log, output_name, fg_name):
    lines = open(input_log).readlines()
    test_log = []
    train_log = []
    startnum = -1
    for line in lines:
        a_line = parsing_line(line)
        if len(a_line):
            if a_line.has_key('test'):
                if a_line['test']:
                    test_log.append(a_line)
                # train_log.append(temp_line)
                # temp_line = a_line
                else:
                    train_log.append(a_line)
            else:
                # logger.info(a_line)
                a = []

    # show image
    show_image(train_log, test_log, output_name, fg_name)


if __name__ == '__main__':
    parser = OptionParser()
    usage = "useage: python %s [options] output_dir" % sys.argv[0]
    parser.add_option("-o", "--output_name", dest="output_name", default='./figure', type="string",
                      help="output_name")
    parser.add_option("-i", "--input_log", dest="input_log", default='', type="string",
                      help="log_name")
    parser.add_option("-n", "--name", dest="name", default='', type="string", help="figure name")
    (options, args) = parser.parse_args()
    main(options.input_log, options.output_name, options.name)
