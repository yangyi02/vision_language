# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:49:37 2016

@author: yangyi05
"""

from __future__ import division
import sys
import os
import io
import math
import logging
# import random
import cPickle
logging.getLogger().setLevel(logging.INFO)


def make_batch(text_file, batch_size, batch_dir):
    """
    This will make batch for the image question answering data
    """
    if not os.path.exists(batch_dir):
        os.mkdir(batch_dir)
    texts = io.open(text_file, 'rb').readlines()
    # Our system assume the first line in the text file correspond to task name
    task = texts[0].strip()
    if task == 'image caption':
        batch_data = batch_image_caption(texts[1:])
    elif task == 'image qa':
        batch_data = batch_image_qa(texts[1:])
    elif task == 'qa':
        batch_data = batch_qa(texts[1:])
    else:
        logging.fatal('Unrecognized task: %s', task)
        sys.exit(1)

    for i in xrange(int(math.ceil(len(batch_data) / batch_size))):
        start_ind = i * batch_size
        end_ind = min((i + 1) * batch_size, len(batch_data))
        data = batch_data[start_ind:end_ind]
        output_file = os.path.join(batch_dir, 'batch_' + str(i))
        cPickle.dump({'task': task, 'data': data}, io.open(output_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        logging.info('Finish batch %d', i)
    logging.info('Finish making batch')

def batch_image_caption(texts):
    """
    Create batch data for image captioning
    """
    batch_data = []
    for line in texts:
        line = line.split('\t')
        if len(line) != 2:
            LOG.fatal('Unrecognized data format for image caption: %s', line)
        data = {}
        data['image_id'] = line[0].strip()
        data['caption'] = line[1].strip()
        batch_data.append(data)
    return batch_data

def batch_image_qa(texts):
    """
    Create batch data for image question answering
    """
    batch_data = []
    for line in texts:
        line = line.split('\t')
        if len(line) != 4:
            LOG.fatal('Unrecognized data format for image qa: %s', line)
        data = {}
        data['image_id'] = line[0].strip()
        data['question_id'] = line[1].strip()
        data['question'] = line[2].strip()
        data['answer'] = line[3].strip()
        batch_data.append(data)
    return batch_data

def batch_qa(texts):
    """
    Create batch data for question answering
    """
    batch_data = []
    for line in texts:
        line = line.split('\t')
        if len(line) != 3:
            LOG.fatal('Unrecognized data format for qa: %s', line)
        data = {}
        data['question_id'] = line[0].strip()
        data['question'] = line[1].strip()
        data['answer'] = line[2].strip()
        batch_data.append(data)
    return batch_data
