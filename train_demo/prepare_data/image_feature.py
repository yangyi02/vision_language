# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:01:31 2016

@author: yangyi05
"""

import os
import sys
import logging
from prepare_data import extract_resnet152, utils
logging.getLogger().setLevel(logging.INFO)


class ImageFeature(object):
    def __init__(self, config):
        self.imageset = config.imageset
        self.image_path = config.image_path
        self.image_feat_type = config.image_feat_type
        self.image_feat_dir = os.path.join('cache/imageset', self.imageset, self.image_feat_type)
        self.image_list = os.path.join(self.image_feat_dir, 'image.list')
        self.image_feat_batch_dir = os.path.join(self.image_feat_dir, 'image_feat_batches')
        self.image_feat_list = os.path.join(self.image_feat_dir, 'image_feat.list')
        if not os.path.exists(self.image_feat_dir):
            os.makedirs(self.image_feat_dir)

    def extract_feature(self):
        """
        Extract image features for a image list.
        TODO: Merge all the image feature extraction code into one file
        """
        if not os.path.exists(self.image_list):
            utils.list_folders(self.image_path, self.image_list)
        logging.info('Image list save to %s', self.image_list)
        if not os.path.exists(self.image_feat_batch_dir):
            if self.image_feat_type == 'resnet152_pool5_2048_oversample':
                extract_resnet152.extract_feature(self.image_list, self.image_feat_batch_dir, oversample=True)
            else:
                logging.fatal('No such feature type: %s', self.image_feat_type)
                sys.exit(1)
        logging.info('Image feature save to %s', self.image_feat_batch_dir)
        if not os.path.exists(self.image_feat_list):
            utils.list_folders(self.image_feat_batch_dir, self.image_feat_list)
        logging.info('Image feature list save to %s', self.image_feat_list)
