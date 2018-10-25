# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import io
import json
import random
import logging
from prepare_data.dataset import DataSetCap
from evaluation.evaluate import EvaluateCap
from show_results.html import HtmlCap
logging.getLogger().setLevel(logging.INFO)


class DataSetInterface(DataSetCap):
    """
    This class defines microsoft coco image captioning dataset.
    """
    def __init__(self, config):
        self.name = 'coco_cap'
        self.use_image = True
        self.imageset = 'coco'
        self.image_path = ['/media/yi/DATA/data-orig/microsoft_coco/coco/images/train2014',
                           '/media/yi/DATA/data-orig/microsoft_coco/coco/images/val2014']
        self.batch_size = 4096
        self.dict_freq_thresh = 0
        self.cache_dir = config.cache_dir
        self.model_dir = config.model_dir
        self.image_feat_type = config.image_feat_type
        self.debug = config.debug
        super(self.__class__, self).__init__()

    def prepare_text(self):
        """
        Prepare text descriptions
        """
        if not os.path.exists(self.train_data_file) or not os.path.exists(self.test_data_file):
            root_path = '/media/yi/DATA/data-orig/microsoft_coco/coco/annotations'
            json_file_train = os.path.join(root_path, 'captions_train2014.json')
            json_file_test = os.path.join(root_path, 'captions_val2014.json')
            data_train = json.load(io.open(json_file_train))
            data_test = json.load(io.open(json_file_test))
            data_train, data_test = self.reorganize_data(data_train, data_test)

            if self.debug: # Batch size is 4096, and we use 3 batches for debugging
                data_train = data_train[:4096*3]
                data_test = data_test[:4096*3]
            self.clean_english_txt(data_train)
            self.clean_english_txt(data_test)
            self.print_to_txt(data_train, self.train_data_file)
            self.print_to_txt(data_test, self.test_data_file)
        logging.info('Training text save to %s', self.train_data_file)
        logging.info('Testing text save to %s', self.test_data_file)

    def reorganize_data(self, data_train, data_test):
        """
        Since there are many testidation data, put them into train data
        """
        logging.info('%d, %d', len(data_train['annotations']), len(data_test['annotations']))
        new_data = []
        image_id_dict = {}
        cnt = 0
        for item in data_test['annotations']:
            image_id = item['image_id']
            if image_id not in image_id_dict:
                data = {}
                data['image_id'] = item['image_id']
                data['id'] = [item['id']]
                data['caption'] = [item['caption']]
                new_data.append(data)
                image_id_dict[image_id] = cnt
                cnt += 1
            else:
                index = image_id_dict[image_id]
                new_data[index]['id'].append(item['id'])
                new_data[index]['caption'].append(item['caption'])
        random.shuffle(new_data)
        new_data_train = self.expand_data(new_data[1024:])
        new_data_test = self.expand_data(new_data[0:1024])
        data_train['annotations'].extend(new_data_train)
        random.shuffle(data_train['annotations'])
        data_test['annotations'] = new_data_test
        logging.info('%d, %d', len(data_train['annotations']), len(data_test['annotations']))
        return data_train['annotations'], data_test['annotations']

    def expand_data(self, new_data):
        """
        new_data is in the format organized by image_id
        expand new_data back to format organized by every sentence
        """
        data_expand = []
        for item in new_data:
            for i in xrange(len(item['caption'])):
                data = {'image_id': item['image_id'], 'id': item['id'][i],
                        'caption': item['caption'][i]}
                data_expand.append(data)
        return data_expand

    def evaluate(self):
        EvaluateCap(self).evaluate()

    def export_html(self):
        html = HtmlCap(self)
        image_ids, urls = self.get_image_url()
        html.print_urls(image_ids, urls)
        html.make_html()
        logging.info('Html results save to %s', html.html_file)

    def get_image_url(self):
        """
        Save image id and url pairs to file
        """
        json_file = '/media/yi/DATA/data-orig/microsoft_coco/coco/annotations/captions_val2014.json'
        data = json.load(io.open(json_file, 'rb'))
        image_ids = [int(sample['id']) for sample in data['images']]
        urls = [sample['flickr_url'] for sample in data['images']]
        return image_ids, urls
