# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import random
import logging
from prepare_data.dataset import DataSetVQA
from show_results.html import HtmlVQA
logging.getLogger().setLevel(logging.INFO)


class DataSetInterface(DataSetVQA):
    """
    This class defines image captioning demo dataset.
    """
    def __init__(self, config):
        self.name = 'demo_vqa'
        self.use_image = True
        self.imageset = 'demo'
        self.image_path = ['demo_images/train', 'demo_images/test']
        self.batch_size = 4096
        self.dict_freq_thresh = 0
        self.cache_dir = config.cache_dir
        self.model_dir = config.model_dir
        self.image_feat_type = config.image_feat_type
        super(self.__class__, self).__init__()

    def prepare_text(self):
        """
        Prepare text descriptions
        """
        train_ids = [2, 7, 9, 15, 16]
        data_train = []
        for i in xrange(len(train_ids)):
            if random.random() < 0.5:
                sample = {'image_id': train_ids[i], 'question_id': train_ids[i] * 10,
                          'question': 'what is in the image', 'answer': 'object'}
            else:
                sample = {'image_id': train_ids[i], 'question_id': train_ids[i] * 10,
                          'question': 'what is in the image', 'answer': 'an object'}
            data_train.append(sample)
        self.print_to_txt(data_train, self.train_data_file)
        logging.info('Training text save to %s', self.train_data_file)

        test_ids = [19, 21, 27, 28, 33]
        data_test = []
        for i in xrange(len(test_ids)):
            if random.random() < 0.5:
                sample = {'image_id': test_ids[i], 'question_id': test_ids[i] * 10,
                          'question': 'what is in the image', 'answer': 'object'}
            else:
                sample = {'image_id': test_ids[i], 'question_id': test_ids[i] * 10,
                          'question': 'what is in the image', 'answer': 'an object'}
            data_test.append(sample)
        self.print_to_txt(data_test, self.test_data_file)
        logging.info('Testing text save to %s', self.test_data_file)

    def export_html(self):
        html = HtmlVQA(self)
        image_ids, urls = self.get_image_url()
        html.print_urls(image_ids, urls)
        html.make_html()
        logging.info('Html results save to %s', html.html_file)

    def get_image_url(self):
        """
        Save image id and url pairs to file
        """
        image_dir = os.path.join(os.getcwd(), 'demo_images/test')
        image_list = os.listdir(image_dir)
        image_ids = [int(image_file.split('.')[0].split('_')[1]) for image_file in image_list]
        urls = [os.path.join(image_dir, image_file) for image_file in image_list]
        return image_ids, urls
