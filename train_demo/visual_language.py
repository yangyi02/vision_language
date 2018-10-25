# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import io
import json
import argparse
import logging
from prepare_data import make_dict, utils
logging.getLogger().setLevel(logging.INFO)


class VisualLanguage(object):
    """
    This class defines the whole process for visual question-answering, captioning, and dialog.
    """
    def __init__(self, config):
        """
        Initialize configuration for training and testing.
        """
        # Global training configuration from ArgumentParser
        self.cache_dir = config.cache_dir
        self.task = config.task
        self.train_conf = config.train_conf
        self.test_conf = config.test_conf
        self.image_feat_type = config.image_feat_type
        self.image_feat_dim = config.image_feat_dim
        self.train_epoch = config.train_epoch
        self.use_gpu = config.use_gpu
        self.trainer_count = config.trainer_count
        # All the intermediate training and testing results will be stored in self.cache_dir
        # The multi-task dictionary file, training and validation list are also stored here.
        self.task_dir = os.path.join(self.cache_dir, 'task', self.task, self.image_feat_type)
        self.image_feat_list = os.path.join(self.task_dir, 'image_feat.list')
        self.word_freq_file = os.path.join(self.task_dir, 'word_freq.txt')
        self.dict_pkl = os.path.join(self.task_dir, 'dict.pkl')
        self.dict_file = os.path.join(self.task_dir, 'dict.txt')
        self.train_list = os.path.join(self.task_dir, 'train.list')
        self.test_list = os.path.join(self.task_dir, 'test.list')
        self.model_name = os.path.splitext(os.path.basename(config.train_conf))[0]
        self.model_dir = os.path.join(self.task_dir, self.model_name)
        self.train_log = os.path.join(self.model_dir, 'train.log')
        # Make directories
        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # Initialize dataset class instances
        self.datasets = []
        for module_name in config.dataset:
            module = __import__(module_name, fromlist=['DataSetInterface'])
            # Add dataset class instance to self.datasets
            self.datasets.append(getattr(module, 'DataSetInterface')(self))

    def prepare_data(self):
        """
        Make multi-task image feature list, word dictionary, training and testing file lists.
        Our task uses multiple language tasks from multiple datasets (caption, qa, dialog).
        """
        for data in self.datasets:
            data.prepare_data()
        self.prepare_feat_list()
        self.prepare_dictionary()
        self.prepare_file_list()

    def prepare_feat_list(self):
        """
        Make file list for image features.
        """
        if not os.path.exists(self.image_feat_list):
            feat_lists = []
            for data in self.datasets:
                if data.use_image:
                    feat_lists.append(data.image_feature.image_feat_list)
            utils.merge_lists(feat_lists, self.image_feat_list)
        logging.info('Multi-task image feature list save to %s', self.image_feat_list)

    def prepare_dictionary(self):
        """
        Make dictionary for multi-task training and testing.
        Our task uses multiple language tasks and multiple datasets (caption, qa, dialog).
        """
        # Compute word frequency for multi-task learning
        if not os.path.exists(self.word_freq_file):
            word_freq_files = [data.word_freq_file for data in self.datasets]
            make_dict.merge_frequency(word_freq_files, self.word_freq_file)
        # Merge dictionaries for multi-task learning
        if not os.path.exists(self.dict_pkl) or not os.path.exists(self.dict_file):
            dict_pkls = [data.dict_pkl for data in self.datasets]
            make_dict.merge_dictionary(dict_pkls, self.dict_pkl, self.dict_file)
        logging.info('Multi-task dictionary save to %s', self.dict_pkl)

    def prepare_file_list(self):
        """
        Make file list for training and testing batches.
        Our task uses multiple dataset and multiple forms of language tasks (caption, qa, dialog).
        """
        train_batch_dirs = [data.train_batch_dir for data in self.datasets]
        utils.list_folders(train_batch_dirs, self.train_list)
        logging.info('Multi-task training file list save to %s', self.train_list)
        test_batch_dirs = [data.test_batch_dir for data in self.datasets]
        utils.list_folders(test_batch_dirs, self.test_list)
        logging.info('Multi-task testing file list save to %s', self.test_list)

    def train(self):
        """
        Train multi-task model
        """
        if not os.path.exists(self.train_log):
            conf_args = 'train_list=%s,test_list=%s,dict_file=%s,dict_pkl=%s,' \
                        'image_feat_list=%s,image_feat_dim=%d' \
                        % (self.train_list, self.test_list, self.dict_file, self.dict_pkl,
                           self.image_feat_list, self.image_feat_dim)
            cmd = 'paddle_trainer --config=%s --use_gpu=%r --trainer_count=%d ' \
                  '--log_period=200 --test_period=0 --num_passes=%d --save_dir=%s ' \
                  '--config_args=%s 2>&1 | tee %s' \
                  % (self.train_conf, self.use_gpu, self.trainer_count,
                     self.train_epoch, self.model_dir, conf_args, self.train_log)
            os.system(cmd)
        logging.info('Model save to %s', self.model_dir)
        logging.info('Train log save to %s', self.train_log)

    def test(self):
        """
        Predict results of multi-task model on multiple tasks
        """
        for data in self.datasets:
            if not os.path.exists(data.result_file):
                conf_args = 'gen_list=%s,result_file=%s,dict_file=%s,dict_pkl=%s,' \
                            'image_feat_list=%s,image_feat_dim=%d,batch_size=%s' \
                            % (data.test_list, data.result_file, self.dict_file, self.dict_pkl,
                               self.image_feat_list, self.image_feat_dim, self.trainer_count)
                cmd = 'paddle_trainer --config=%s --use_gpu=%r --trainer_count=%d ' \
                      '--log_period=200 --test_period=0 --num_passes=%d --save_dir=%s ' \
                      '--job=test --test_pass=%d --config_args=%s' \
                      % (self.test_conf, self.use_gpu, self.trainer_count,
                         self.train_epoch, self.model_dir, self.train_epoch - 1, conf_args)
                os.system(cmd)
            logging.info('Prediction results save to %s', data.result_file)

    def export_html(self):
        """
        Generate html from prediction results
        """
        for data in self.datasets:
            data.export_html()

    def run(self):
        """
        Main function for training, testing and demo.
        """
        self.prepare_data()
        self.train()
        self.test()
        self.export_html()
        logging.info('Done')


if __name__ == '__main__':
    """
    Main function for training and testing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', help='directory storing all cache files', default='cache')
    parser.add_argument('--task', help='training task', default='')
    parser.add_argument('--dataset', nargs='+', help='training dataset', default='')
    parser.add_argument('--train_conf', help='training config', default='')
    parser.add_argument('--test_conf', help='prediction config', default='')
    parser.add_argument('--image_feat_type', help='image feature type', default='')
    parser.add_argument('--image_feat_dim', help='image feature dimension', type=int, default=0)
    parser.add_argument('--train_epoch', help='training epoch', type=int, default=100)
    parser.add_argument('--use_gpu', help='enable gpu mode', action='store_true', default=False)
    parser.add_argument('--trainer_count', help='number of gpus/cpus', type=int, default=1)
    args = parser.parse_args()

    visual_language = VisualLanguage(args)
    visual_language.run()
