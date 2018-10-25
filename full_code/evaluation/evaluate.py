# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import io
import json
import logging
from evaluation import eval
logging.getLogger().setLevel(logging.INFO)


class Evaluate(object):
    def __init__(self, dataset):
        self.result_file = dataset.result_file
        self.test_data_file = dataset.test_data_file
        self.gt_json = os.path.join(dataset.data_dir, 'groundtruth.json')
        self.result_json = os.path.join(dataset.model_dir, 'result_' + dataset.name + '.json')
        self.score_json = os.path.join(dataset.model_dir, 'score_' + dataset.name + '.json')

    def prepare_predict_json(self):
        # Make prediction json for evaluation usage
        if not os.path.exists(self.result_json):
            self.make_predict_json(self.result_file, self.result_json)
        logging.info('Result json save to %s', self.result_json)

    def prepare_gt_json(self):
        # Make ground truth json for evaluation usage
        if not os.path.exists(self.gt_json):
            self.make_gt_json(self.test_data_file, self.gt_json)
        logging.info('Ground truth json save to %s', self.gt_json)


class EvaluateCap(Evaluate):
    def evaluate(self):
        """
        Evaluation code
        """
        self.prepare_predict_json()
        self.prepare_gt_json()
        if not os.path.exists(self.score_json):
            eval.evaluate_cap(self.result_json, self.gt_json, self.score_json)
        logging.info('Benchmark scores save to %s', self.score_json)

    def make_gt_json(self, text_file, gt_json):
        """
        Make ground truth json file for evaluation usage
        """
        data = []
        lines = io.open(text_file).readlines()
        lines = lines[1:]
        for line in lines:
            items = line.strip().split('\t')
            sample = {'image_id': int(items[0]), 'caption': items[1]}
            data.append(sample)
        json.dump(data, io.open(gt_json, 'wb'))

    def make_predict_json(self, text_file, json_file):
        """
        Make caption results into a json results for benchmark evaluation
        """
        image_ids, questions, answers = self.read_text(text_file)
        question_dict = dict(zip(image_ids, questions))
        answer_dict = dict(zip(image_ids, answers))
        image_ids = list(set(image_ids))  # There are duplicate image ids
        res = []
        for image_id in image_ids:
            entry = {'image_id': int(image_id), 'caption': answer_dict[image_id]}
            res.append(entry)
        json.dump(res, io.open(json_file, 'wb'))

    def read_text(self, text_file):
        """
        Load result text file, the text file consists of 3 columns:
        id, question, answer
        """
        data = io.open(text_file, 'rb').readlines()
        image_ids, questions, answers = [], [], []
        for line in data:
            items = line.strip().split('\t')
            image_ids.append(items[0].strip())
            questions.append(items[1].strip())
            # Remove end sign in the prediction answers
            answers.append(items[2].replace('$$E$$', '').strip())
        return image_ids, questions, answers


class EvaluateVQA(Evaluate):
    def evaluate(self):
        """
        Evaluation code
        """
        # Make prediction json for evaluation usage
        self.prepare_predict_json()
        self.prepare_gt_json()
        if not os.path.exists(self.score_json):
            eval.evaluate_qa(self.result_json, self.gt_json, self.score_json)
        logging.info('Benchmark scores save to %s', self.score_json)

    def make_gt_json(self, text_file, gt_json):
        """
        Make ground truth json file for evaluation usage
        """
        data = []
        lines = io.open(self.test_data_file).readlines()
        lines = lines[1:]
        for line in lines:
            items = line.strip().split('\t')
            sample = {'image_id': int(items[0]), 'question_id': int(items[1]), 'question': items[2],
                      'answer': items[3]}
            data.append(sample)
        json.dump(data, io.open(gt_json, 'wb'))

    def make_predict_json(self, text_file, json_file):
        """
        Make QA results into a json results for benchmark evaluation
        """
        question_ids, questions, answers = self.read_text(text_file)
        question_dict = dict(zip(question_ids, questions))
        answer_dict = dict(zip(question_ids, answers))
        res = []
        for question_id in question_ids:
            entry = {'question_id': int(question_id), 'answer': answer_dict[question_id]}
            res.append(entry)
        json.dump(res, io.open(json_file, 'wb'))

    def read_text(self, text_file):
        """
        Load result text file, the text file consists of 3 columns:
        id, question, answer
        """
        data = io.open(text_file, 'rb').readlines()
        question_ids, questions, answers = [], [], []
        for line in data:
            items = line.strip().split('\t')
            question_ids.append(items[0].strip())
            questions.append(items[1].strip())
            # Remove end sign in the prediction answers
            answers.append(items[2].replace('$$E$$', '').strip())
        return question_ids, questions, answers


class EvaluateQA(Evaluate):
    def evaluate(self):
        """
        Evaluation code
        """
        # Make prediction json for evaluation usage
        self.prepare_predict_json()
        self.prepare_gt_json()
        if not os.path.exists(self.score_json):
            eval.evaluate_qa(self.result_json, self.gt_json, self.score_json)
        logging.info('Benchmark scores save to %s', self.score_json)

    def make_gt_json(self, text_file, gt_json):
        """
        Make ground truth json file for evaluation usage
        """
        data = []
        lines = io.open(text_file).readlines()
        lines = lines[1:]
        for line in lines:
            items = line.strip().split('\t')
            sample = {'question_id': int(items[0]), 'question': items[1], 'answer': items[2]}
            data.append(sample)
        json.dump(data, io.open(gt_json, 'wb'))

    def make_predict_json(self, text_file, json_file):
        """
        Make QA results into a json results for benchmark evaluation
        """
        question_ids, questions, answers = self.read_text(text_file)
        question_dict = dict(zip(question_ids, questions))
        answer_dict = dict(zip(question_ids, answers))
        res = []
        for question_id in question_ids:
            entry = {'question_id': int(question_id), 'answer': answer_dict[question_id]}
            res.append(entry)
        json.dump(res, io.open(json_file, 'wb'))

    def read_text(self, text_file):
        """
        Load result text file, the text file consists of 3 columns:
        id, question, answer
        """
        data = io.open(text_file, 'rb').readlines()
        question_ids, questions, answers = [], [], []
        for line in data:
            items = line.strip().split('\t')
            question_ids.append(items[0].strip())
            questions.append(items[1].strip())
            # Remove end sign in the prediction answers
            answers.append(items[2].replace('$$E$$', '').strip())
        return question_ids, questions, answers
