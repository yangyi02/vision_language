# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import io
import logging
logging.getLogger().setLevel(logging.INFO)


class Html(object):
    def __init__(self, config):
        self.url_file = os.path.join(config.data_dir, 'url.txt')
        self.test_data_file = config.test_data_file
        self.result_file = config.result_file
        self.html_file = os.path.join(config.model_dir, config.name + '.html')

    def print_urls(self, image_ids, urls):
        """
        Save image id and image local url pairs
        """
        handle = io.open(self.url_file, 'wb')
        for image_id, url in zip(image_ids, urls):
            handle.write('%d\t%s\n' % (image_id, url))

    def read_urls(self):
        """
        Load image id and image url pairs
        """
        data = io.open(self.url_file, 'rb').readlines()
        image_ids, image_urls = [], []
        for line in data:
            items = line.strip().split('\t')
            image_ids.append(int(items[0]))
            image_urls.append(items[1])
        return image_ids, image_urls

    def read_result(self):
        """
        Load result text file, the text file consists of 3 columns: id, question, answer
        """
        data = io.open(self.result_file, 'rb').readlines()
        question_ids, questions, answers = [], [], []
        for line in data:
            items = line.strip().split('\t')
            question_ids.append(int(items[0].strip()))
            questions.append(items[1].strip())
            # Remove end sign in the prediction answers
            answers.append(items[2].replace('$$E$$', '').strip())
        return question_ids, questions, answers


class HtmlCap(Html):
    def make_html(self):
        """
        Create html file
        """
        image_ids, image_urls = self.read_urls()
        url_dict = dict(zip(image_ids, image_urls))

        image_ids, questions, captions = self.read_result()
        caption_dict = dict(zip(image_ids, captions))

        image_ids, gt_captions = self.read_ground_truth()
        gt_caption_dict = dict(zip(image_ids, gt_captions))

        handle = io.open(self.html_file, 'wb')
        handle.write('<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<html>\n<body>\n')
        handle.write('<h1>Image Caption</h1>\n<table style=\"width:80%\" cellpadding=\"10\">\n')
        handle.write('<tr><td>Image</td><td>Caption</td><td>Ground truth</td></tr>\n')
        for image_id in list(set(image_ids)):
            if image_id in url_dict:
                handle.write(self.wrap_image_text(url_dict[image_id], caption_dict[image_id],
                                                  gt_caption_dict[image_id]))
        handle.write('</table>\n</body>\n</html>\n')
        logging.info('Html results save to %s', self.html_file)

    @staticmethod
    def wrap_image_text(url, caption, gt_caption):
        """
        Create an image text pair in html
        """
        text = '<tr><td><img src=\"' + url + '\" style=\"width:256px\"></td>'
        text += '<td>' + caption.strip() + '</td>'
        text += '<td>' + gt_caption.strip() + '</td></tr>\n'
        return text

    def read_ground_truth(self):
        """
        Load ground truth text file
        """
        data = io.open(self.test_data_file, 'rb').readlines()
        data = data[1:]
        image_ids, captions = [], []
        for line in data:
            items = line.strip().split('\t')
            image_ids.append(int(items[0].strip()))
            captions.append(items[1].strip())
        return image_ids, captions


class HtmlVQA(Html):
    def make_html(self):
        """
        Create html file
        """
        image_ids, image_urls = self.read_urls()
        url_dict = dict(zip(image_ids, image_urls))

        question_ids, questions, answers = self.read_result()
        question_dict = dict(zip(question_ids, questions))
        answer_dict = dict(zip(question_ids, answers))

        image_ids, question_ids, questions, gt_answers = self.read_ground_truth()
        gt_answer_dict = dict(zip(question_ids, gt_answers))
        question_to_image = dict(zip(question_ids, image_ids))

        handle = io.open(self.html_file, 'wb')
        handle.write('<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<html>\n<body>\n')
        handle.write('<h1>Image QA</h1>\n<table style=\"width:80%\" cellpadding=\"10\">\n')
        handle.write('<tr><td>Image</td><td>Question</td><td>Answer</td><td>Ground truth</td></tr>\n')
        for question_id in list(set(question_ids)):
            image_id = question_to_image[question_id]
            if image_id in url_dict:
                handle.write(self.wrap_image_text(url_dict[image_id], question_dict[question_id],
                                                  answer_dict[question_id], gt_answer_dict[question_id]))
        handle.write('</table>\n</body>\n</html>\n')
        logging.info('Html results save to %s', self.html_file)

    @staticmethod
    def wrap_image_text(url, question, answer, gt_answer):
        """
        Create an image text pair in html
        """
        text = '<tr><td><img src=\"' + url + '\" style=\"width:256px\"></td>'
        text += '<td>' + question.strip() + '</td>'
        text += '<td>' + answer.strip() + '</td>'
        text += '<td>' + gt_answer.strip() + '</td></tr>\n'
        return text

    def read_ground_truth(self):
        """
        Load ground truth text file
        """
        data = io.open(self.test_data_file, 'rb').readlines()
        data = data[1:]
        image_ids, question_ids, questions, answers = [], [], [], []
        for line in data:
            items = line.strip().split('\t')
            image_ids.append(int(items[0].strip()))
            question_ids.append(int(items[1].strip()))
            questions.append(items[2].strip())
            answers.append(items[3].strip())
        return image_ids, question_ids, questions, answers


class HtmlQA(Html):
    def make_html(self):
        """
        Create html file
        """
        question_ids, questions, answers = self.read_result()
        question_dict = dict(zip(question_ids, questions))
        answer_dict = dict(zip(question_ids, answers))

        question_ids, questions, gt_answers = self.read_ground_truth()
        gt_answer_dict = dict(zip(question_ids, gt_answers))

        handle = io.open(self.html_file, 'wb')
        handle.write('<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<html>\n<body>\n')
        handle.write('<h1>QA</h1>\n<table style=\"width:80%\" cellpadding=\"10\">\n')
        handle.write('<tr><td>Image</td><td>Question</td><td>Answer</td><td>Ground truth</td></tr>\n')
        for question_id in list(set(question_ids)):
            handle.write(self.wrap_image_text(question_dict[question_id], answer_dict[question_id],
                                              gt_answer_dict[question_id]))
        handle.write('</table>\n</body>\n</html>\n')
        logging.info('Html results save to %s', self.html_file)

    @staticmethod
    def wrap_image_text(question, answer, gt_answer):
        """
        Create a question answer pair in html
        """
        text = '<tr><td>' + question.strip() + '</td>'
        text += '<td>' + answer.strip() + '</td>'
        text += '<td>' + gt_answer.strip() + '</td></tr>\n'
        return text

    def read_ground_truth(self):
        """
        Load ground truth text file
        """
        data = io.open(self.test_data_file, 'rb').readlines()
        data = data[1:]
        question_ids, questions, answers = [], [], []
        for line in data:
            items = line.strip().split('\t')
            question_ids.append(int(items[0].strip()))
            questions.append(items[1].strip())
            answers.append(items[2].strip())
        return question_ids, questions, answers
