# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import io
import logging
from prepare_data import make_dict, make_batch, utils
from image_feature import ImageFeature
from prepare_data.clean_sentence import process_english_sentence, process_chinese_sentence
logging.getLogger().setLevel(logging.INFO)


class DataSet(object):
    """
    This class specify parameters for language data preparation, training and testing
    """
    def __init__(self):
        self.data_dir = os.path.join(self.cache_dir, 'dataset', self.name)
        self.data_dir += '_debug' if self.debug else ''
        self.train_data_file = os.path.join(self.data_dir, 'train_data.txt')
        self.test_data_file = os.path.join(self.data_dir, 'test_data.txt')
        self.word_freq_file = os.path.join(self.data_dir, 'word_freq.txt')
        self.dict_pkl = os.path.join(self.data_dir, 'dict.pkl')
        self.dict_file = os.path.join(self.data_dir, 'dict.txt')
        self.train_batch_dir = os.path.join(self.data_dir, 'train_batches')
        self.test_batch_dir = os.path.join(self.data_dir, 'test_batches')
        self.train_list = os.path.join(self.data_dir, 'train.list')
        self.test_list = os.path.join(self.data_dir, 'test.list')
        self.result_file = os.path.join(self.model_dir, 'result_' + self.name + '.txt')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if self.use_image:
            self.image_feature = ImageFeature(self)

    def prepare_data(self):
        """
        Prepare image features, word dictionary, training and testing batches for a dataset.
        """
        # Prepare image features for every dataset given a pre-trained image recognition model.
        self.prepare_feat()
        # Prepare training and testing text files for every dataset.
        self.prepare_text()
        # Prepare word dictionary for every dataset.
        self.prepare_dictionary()
        # Prepare batches that contain image id and natural languages description for every dataset.
        self.prepare_batch()
        # Prepare training and testing language file list for each dataset.
        utils.list_folders(self.train_batch_dir, self.train_list)
        utils.list_folders(self.test_batch_dir, self.test_list)

    def prepare_feat(self):
        """
        Prepare image features files for training and testing.
        """
        if self.use_image:
            self.image_feature.extract_feature()

    def prepare_text(self):
        """
        Prepare training and testing text files.
        """
        logging.error('You should have prepare_text function specified in each data module!')

    def prepare_dictionary(self):
        """
        Make word dictionary as well as compute word frequency count.
        """
        if not os.path.exists(self.word_freq_file):
            data = io.open(self.train_data_file, 'rb').readlines()
            make_dict.compute_frequency(data, self.word_freq_file)
        if not os.path.exists(self.dict_pkl) or not os.path.exists(self.dict_file):
            make_dict.make_dictionary(self.word_freq_file, self.dict_freq_thresh, self.dict_pkl,
                                      self.dict_file)
        logging.info('Data dictionary save to %s', self.dict_pkl)

    def prepare_batch(self):
        """
        Make training and testing batches which contain image ids and texts.
        Note that image features are stored in another image feat batch directory.
        The word dictionary is also stored in a dictionary pickle file.
        During training and testing, dictionary and all image features will be loaded into memory.
        This will save a lot of disks because usually one image can correspond to multiple texts.
        """
        if not os.path.exists(self.train_batch_dir):
            make_batch.make_batch(self.train_data_file, self.batch_size, self.train_batch_dir)
        logging.info('Training data batch save to %s', self.train_batch_dir)
        if not os.path.exists(self.test_batch_dir):
            make_batch.make_batch(self.test_data_file, self.batch_size, self.test_batch_dir)
        logging.info('Testing data batch save to %s', self.test_batch_dir)


class DataSetCap(DataSet):
    def clean_english_txt(self, data):
        for sample in data:
            sample['caption'] = process_english_sentence(sample['caption'])
    
    def clean_chinese_text(self, data):    
        for sample in data:
            sample['caption'] = process_chinese_sentence(sample['caption'])
    
    def print_to_txt(self, data, output_file):
        """
        Print image caption data to output text file
        """
        handle = io.open(output_file, 'wb')
        handle.write('image caption\n')
        for sample in data:
            image_id = sample['image_id']
            caption = sample['caption']
            handle.write('%d\t%s\n' % (image_id, caption))


class DataSetVQA(DataSet):
    def clean_english_txt(self, data):
        for sample in data:
            sample['question'] = process_english_sentence(sample['question'])
            sample['answer'] = process_english_sentence(sample['answer'])
    
    def clean_chinese_text(self, data):    
        for sample in data:
            sample['question'] = process_chinese_sentence(sample['question'])
            sample['answer'] = process_chinese_sentence(sample['answer'])
    
    def print_to_txt(self, data, output_file):
        """
        Print image question answer data to output text file
        """
        handle = io.open(output_file, 'wb')
        handle.write('image qa\n')
        for sample in data:
            image_id = sample['image_id']
            question_id = sample['question_id']
            question = sample['question']
            answer = sample['answer']
            handle.write('%d\t%d\t%s\t%s\n' % (image_id, question_id, question, answer))


class DataSetQA(DataSet):
    def clean_english_txt(self, data):
        for sample in data:
            sample['question'] = process_english_sentence(sample['question'])
            sample['answer'] = process_english_sentence(sample['answer'])
    
    def clean_chinese_text(self, data):    
        for sample in data:
            sample['question'] = process_chinese_sentence(sample['question'])
            sample['answer'] = process_chinese_sentence(sample['answer'])
    
    def print_to_txt(self, data, output_file):
        """
        Print question answer data to output text file
        """
        handle = io.open(output_file, 'wb')
        handle.write('qa\n')
        for sample in data:
            question_id = sample['question_id']
            question = sample['question']
            answer = sample['answer']
            handle.write('%d\t%s\t%s\n' % (question_id, question, answer))
