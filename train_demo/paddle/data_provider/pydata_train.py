# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:42:19 2016

@author: yangyi05
"""

import sys
import os
import io
# import time
import cPickle
import logging
import numpy
import random
from paddle.trainer.PyDataProviderWrapper import DenseSlot, IndexSlot, provider

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def initHook(obj, *file_list, **kwargs):
    """
    Description: Init with a list of data file
    file_list is the name list of input files.
    kwargs['load_data_args'] is the value of 'load_data_args'
    which can be set in config.
    kwargs['load_data_args'] is organized as follows:
        'dictionary path'(str)
        'image feature list file'(str)
        'img_feat_dim'(int)
        'average norm factor for image feature'(float)
    Each args is seperated by a space.
    """
    str_conf_args = kwargs['load_data_args'].strip().split()
    dict_file = str_conf_args[0]
    img_feat_list = str_conf_args[1]
    img_feat_dim = int(str_conf_args[2])
    feat_avg_norm_factor = float(str_conf_args[3])

    LOG.info('Dictionary path: %s', dict_file)
    LOG.info('Image feature list: %s', img_feat_list)
    LOG.info('Image dimension: %d', img_feat_dim)
    LOG.info('Image feature norm factor: %.4f', feat_avg_norm_factor)

    if os.path.isfile(dict_file):
        word_dict = cPickle.load(io.open(dict_file, 'rb'))
        if word_dict.get('#OOV#', -1) == -1:
            word_dict['#OOV#'] = 0
        if word_dict.get('$$S$$', -1) == -1:
            word_dict['$$S$$'] = len(word_dict)
        if word_dict.get('$$E$$', -1) == -1:
            word_dict['$$E$$'] = len(word_dict)
        LOG.info('Dictionary loaded with %d words', len(word_dict))
    else:
        LOG.fatal('Dictionary file %s does not exist!', dict_file)
        sys.exit(1)

    if len(file_list) == 0:
        LOG.fatal('No annotation file!')
        sys.exit(1)
    else:
        LOG.info('There are %d annotation files', len(file_list))
    file_name = file_list[0].strip()
    if os.path.isfile(file_name):
        LOG.debug('Annotation file name: %s', file_name)
    else:
        LOG.fatal('Annotation file %s missing!', file_name)
        sys.exit(1)

    if os.path.exists(img_feat_list):
        img_feat_list = io.open(img_feat_list, 'rb').readlines()
        if len(img_feat_list) == 0:
            LOG.fatal('No image feature file!')
            sys.exit(1)
        else:
            LOG.info('There are %d feature files', len(img_feat_list))
    else:
        LOG.fatal('Image feature list %s does not exist!', img_feat_list)
        sys.exit(1)
    file_name = img_feat_list[0].strip()
    if os.path.isfile(file_name):
        LOG.debug('Image feature file name: %s', file_name)
    else:
        LOG.fatal('Image feature file %s missing!', file_name)
        sys.exit(1)

    obj.file_list = list(file_list)
    obj.word_dict = word_dict
    obj.features = load_image_feature(img_feat_list)
    obj.img_feat_dim = img_feat_dim
    obj.feat_avg_norm_factor = feat_avg_norm_factor
    LOG.info('DataProvider Initialization finished')
    obj.slots = [DenseSlot(img_feat_dim), IndexSlot(len(word_dict)),
                 IndexSlot(len(word_dict)), IndexSlot(len(word_dict))]

def load_image_feature(file_list):
    """
    Load image feature
    """
    features = {}
    for file_name in file_list:
        logging.info('Load feature file %s', file_name.strip())
        feature = cPickle.load(io.open(file_name.strip(), 'rb'))
        features.update(feature)
    return features

@provider(use_seq=True, init_hook=initHook)
def processData(obj, file_name):
    """
    Description: Get a batch of samples
    Return format:
        image feature, question coding, current word coding, next word coding
    """
    # begin_time = time.time()
    data = cPickle.load(io.open(file_name, 'rb'))
    task = data['task']
    random.shuffle(data['data'])
    for sample in data['data']:
        if task == 'image caption':
            question_id, img_feat, coding_q, coding_a = get_data_image_caption(obj, sample)
        elif task == 'image qa':
            question_id, img_feat, coding_q, coding_a = get_data_image_qa(obj, sample)
        elif task == 'qa':
            question_id, img_feat, coding_q, coding_a = get_data_qa(obj, sample)
        else:
            LOG.fatal('Unrecognized task: %s', task)
            sys.exit(1)
        if img_feat is None:
            continue
        if coding_a[0] != obj.word_dict.get('$$S$$', -1):
            coding_a.insert(0, obj.word_dict.get('$$S$$', -1))
        if coding_a[-1] != obj.word_dict.get('$$E$$', -1):
            coding_a.append(obj.word_dict.get('$$E$$', -1))
        coding_current_word = coding_a[:-1]
        coding_next_word = coding_a[1:]
        yield [img_feat], coding_q, coding_current_word, coding_next_word
    # end_time = time.time()
    # LOG.info('Loading time spent: %f seconds', end_time - begin_time)

def get_data_image_caption(obj, sample):
    """
    Get data from a image caption sample
    """
    caption = sample['caption'].strip().split(' ')
    coding_q = [0]
    coding_a = [obj.word_dict.get(word, 0) for word in caption]
    img_feat = get_image_feature(obj.features, sample['image_id'])
    if img_feat is not None:
        img_feat = img_feat * obj.feat_avg_norm_factor
    question_id = int(sample['image_id'])
    return question_id, img_feat, coding_q, coding_a

def get_data_image_qa(obj, sample):
    """
    Get data from a image qa sample
    """
    question = sample['question'].strip().split(' ')
    answer = sample['answer'].strip().split(' ')
    coding_q = [obj.word_dict.get(word, 0) for word in question]
    coding_a = [obj.word_dict.get(word, 0) for word in answer]
    img_feat = get_image_feature(obj.features, sample['image_id'])
    if img_feat is not None:
        img_feat = img_feat * obj.feat_avg_norm_factor
    question_id = int(sample['question_id'])
    return question_id, img_feat, coding_q, coding_a

def get_data_qa(obj, sample):
    """
    Get data from a qa sample
    """
    question = sample['question'].strip().split(' ')
    answer = sample['answer'].strip().split(' ')
    coding_q = [obj.word_dict.get(word, 0) for word in question]
    coding_a = [obj.word_dict.get(word, 0) for word in answer]
    img_feat = numpy.zeros((obj.img_feat_dim,), dtype=numpy.float32)
    question_id = int(sample['question_id'])
    return question_id, img_feat, coding_q, coding_a

def get_image_feature(feature_dict, image_id):
    """
    Get image feature from feature dictionary
    """
    if image_id in feature_dict:
        return feature_dict[image_id]
    elif str(image_id) in feature_dict:
        return feature_dict[str(image_id)]
    elif int(image_id) in feature_dict:
        return feature_dict[int(image_id)]
    else:
        return None
