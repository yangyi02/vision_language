# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:42:19 2015

@author: yangyi05
"""

import os
import io
import time
import cPickle
import logging
import numpy
from paddle.trainer.PyDataProviderWrapper import DenseSlot, IndexSlot, provider

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('paddle_data_provider')
logger.setLevel(logging.INFO)

def initHook(obj, *file_list, **kwargs):
    '''
    Description: Init with a list of data file
    file_list is the name list of input files.
    kwargs["load_data_args"] is the value of 'load_data_args'
    which can be set in config.
    kwargs["load_data_args"] is organized as follows:
        'dictionary path'(str)
        'img_feat_dim'(int)
        'average norm factor for image feature'(float)
    Each args is seperated by a space.
    '''
    str_conf_args = kwargs['load_data_args'].strip().split()
    dict_file = str_conf_args[0]
    img_feat_dim = int(str_conf_args[1])
    feat_avg_norm_factor = float(str_conf_args[2])

    logger.info('Dictionary Path: %s', dict_file)
    logger.info('Image dimension: %d', img_feat_dim)
    logger.info('Average norm factor for image feature: %.4f',
                feat_avg_norm_factor)

    if os.path.isfile(dict_file):
        word_dict = cPickle.load(io.open(dict_file, 'rb'))
        if word_dict.get('$$S$$', -1) == -1:
            word_dict['$$S$$'] = len(word_dict)
        if word_dict.get('$$E$$', -1) == -1:
            word_dict['$$E$$'] = len(word_dict)
        logger.info('Dictionary loaded with %d words', len(word_dict))
    else:
        logger.fatal('Dictionary file [%s] does not exist!', dict_file)

    if len(file_list) == 0:
        logger.fatal('No annotation file!')
    else:
        logger.info('There are %d annotation files', len(file_list))

    fname = file_list[0].strip()
    if os.path.isfile(fname):
        logger.debug("fname %s", fname)
    else:
        logger.warn('Annotation file %s missing!', fname)

    obj.file_list = list(file_list)
    obj.word_dict = word_dict
    obj.img_feat_dim = img_feat_dim
    obj.feat_avg_norm_factor = feat_avg_norm_factor
    logger.info('DataProvider Initialization finished')
    obj.slots = [IndexSlot(1), DenseSlot(img_feat_dim), IndexSlot(len(word_dict))]

@provider(use_seq=True, init_hook=initHook)
def processData(obj, file_name):
    '''
    Description: Get a batch of samples
    Return format:
        image feature, question coding, current word coding, next word coding
    '''
    begin_time = time.time()
    word_dict = obj.word_dict
    data = cPickle.load(io.open(file_name, 'rb'))
    for sample in data:
        if 'sen' in sample:
            coding_q = [0]
            coding_a = [word_dict.get(word, 0) for word in sample['sen']]
        elif 'question' in sample:
            coding_q = [word_dict.get(word, 0) for word in sample['question']]
            coding_a = [word_dict.get(word, 0) for word in sample['answer']]
        else:
            logger.fatal('Error Data Format')
        if len(sample['feature']) == 0:
            img_feat = numpy.zeros((obj.img_feat_dim,), dtype=numpy.float32)
        else:
            img_feat = sample['feature'] * obj.feat_avg_norm_factor
        if len(sample['question_id']) > 0:
            question_id = int(sample['question_id'])
        elif len(sample['image_id']) > 0:
            question_id = int(sample['image_id'])
        else:
            logger.fatal('Must specify an id for generating texts!')
        yield [question_id], [img_feat], coding_q
    end_time = time.time()
    # logger.info('Loading time spent: %f seconds', end_time - begin_time)
