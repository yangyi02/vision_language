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
from prepare_data.dataset import DataSetVQA
from evaluation.evaluate import EvaluateVQA
from show_results.html import HtmlVQA

logging.getLogger().setLevel(logging.INFO)


class DataSetInterface(DataSetVQA):
    """
    This class defines microsoft coco image captioning dataset.
    """
    def __init__(self, config):
        self.name = 'coco_vqa'
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
            root_path_q = '/media/yi/DATA/data-orig/microsoft_coco/VQA/Questions'
            root_path_a = '/media/yi/DATA/data-orig/microsoft_coco/VQA/Annotations'
            json_file_train_q = os.path.join(root_path_q,
                                             'OpenEnded_mscoco_train2014_questions.json')
            json_file_train_a = os.path.join(root_path_a, 'mscoco_train2014_annotations.json')
            json_file_test_q = os.path.join(root_path_q, 'OpenEnded_mscoco_val2014_questions.json')
            json_file_test_a = os.path.join(root_path_a, 'mscoco_val2014_annotations.json')
            data_train_q = json.load(io.open(json_file_train_q))
            data_train_a = json.load(io.open(json_file_train_a))
            data_test_q = json.load(io.open(json_file_test_q))
            data_test_a = json.load(io.open(json_file_test_a))
            data_train, data_test = \
                self.reorganize_data(data_train_q, data_train_a, data_test_q, data_test_a)

            if self.debug:  # Batch size is 4096, and we use 3 batches for debugging
                data_train = data_train[:4096 * 3]
                data_test = data_test[:4096 * 3]
            self.clean_english_txt(data_train)
            self.clean_english_txt(data_test)
            self.print_to_txt(data_train, self.train_data_file)
            self.print_to_txt(data_test, self.test_data_file)
        logging.info('Training text save to %s', self.train_data_file)
        logging.info('Testing text save to %s', self.test_data_file)

    def reorganize_data(self, data_train_q, data_train_a, data_test_q, data_test_a):
        """
        Since there are many validation data, put them into train data
        """
        logging.info('%d, %d, %d, %d', len(data_train_q['questions']), len(data_train_a['annotations']),
                     len(data_test_q['questions']), len(data_test_a['annotations']))
        # question_answer_pair = zip(data_test_q['questions'], data_test_a['annotations'])
        data_train = {'annotations': data_train_q['questions']}
        for i in xrange(len(data_train_q['questions'])):
            data_train['annotations'][i]['answer'] = data_train_a['annotations'][i][
                'multiple_choice_answer']
        data_test = {'annotations': data_test_q['questions']}
        for i in xrange(len(data_test_q['questions'])):
            data_test['annotations'][i]['answer'] = data_test_a['annotations'][i][
                'multiple_choice_answer']
        new_data = []
        image_id_dict = {}
        cnt = 0
        for item in data_test['annotations']:
            image_id = item['image_id']
            if image_id not in image_id_dict:
                data = {'image_id': item['image_id'], 'question_id': [item['question_id']],
                        'question': [item['question']], 'answer': [item['answer']]}
                new_data.append(data)
                image_id_dict[image_id] = cnt
                cnt += 1
            else:
                index = image_id_dict[image_id]
                new_data[index]['question_id'].append(item['question_id'])
                new_data[index]['question'].append(item['question'])
                new_data[index]['answer'].append(item['answer'])
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
            for i in xrange(len(item['question'])):
                data = {'image_id': item['image_id'], 'question_id': item['question_id'][i],
                        'question': item['question'][i], 'answer': item['answer'][i]}
                data_expand.append(data)
        return data_expand

    def evaluate(self):
        EvaluateVQA(self).evaluate()
        # Below is speical to microsoft coco vqa evaluation
        root_path_q = '/media/yi/DATA/data-orig/microsoft_coco/VQA/Questions'
        root_path_a = '/media/yi/DATA/data-orig/microsoft_coco/VQA/Annotations'
        ques_json = os.path.join(root_path_q, 'OpenEnded_mscoco_val2014_questions.json')
        anno_json = os.path.join(root_path_a, 'mscoco_val2014_annotations.json')
        from evaluation import eval
        score_json = os.path.join(self.model_dir, 'coco_vqa_score.json')
        eval.evaluate_coco_vqa(EvaluateVQA(self).result_json, ques_json, anno_json, score_json)
        logging.info('Coco VQA special evaluation score save to %s', score_json)

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
        json_file = '/media/yi/DATA/data-orig/microsoft_coco/coco/annotations/captions_val2014.json'
        data = json.load(io.open(json_file, 'rb'))
        image_ids = [int(sample['id']) for sample in data['images']]
        urls = [sample['flickr_url'] for sample in data['images']]
        return image_ids, urls

