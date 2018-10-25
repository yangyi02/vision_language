# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import random
import logging
from prepare_data.dataset import DataSetQA
from show_results.html import HtmlQA
logging.getLogger().setLevel(logging.INFO)


class DataSetInterface(DataSetQA):
    """
    This class defines image captioning demo dataset.
    """
    def __init__(self, config):
        self.name = 'demo_qa'
        self.use_image = False
        self.batch_size = 4096
        self.dict_freq_thresh = 0
        self.cache_dir = config.cache_dir
        self.model_dir = config.model_dir
        super(self.__class__, self).__init__()

    def prepare_text(self):
        """
        Prepare text descriptions
        """
        data_train = []
        for i in xrange(5):
            if random.random() < 0.5:
                sample = {'question_id': i, 'question': 'what is in the image', 'answer': 'object'}
            else:
                sample = {'question_id': i, 'question': 'what is in the image',
                          'answer': 'an object'}
            data_train.append(sample)
        self.print_to_txt(data_train, self.train_data_file)
        logging.info('Training text save to %s', self.train_data_file)

        data_test = []
        for i in xrange(5):
            if random.random() < 0.5:
                sample = {'question_id': i, 'question': 'what is in the image', 'answer': 'object'}
            else:
                sample = {'question_id': i, 'question': 'what is in the image',
                          'answer': 'an object'}
            data_test.append(sample)
        self.print_to_txt(data_test, self.test_data_file)
        logging.info('Training text save to %s', self.test_data_file)

    def export_html(self):
        html = HtmlQA(self)
        html.make_html()
