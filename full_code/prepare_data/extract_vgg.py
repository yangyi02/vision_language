# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:02:23 2015

@author: yangyi05
"""

from __future__ import division
import sys
import os
from os import path
import io
import logging
import cPickle
import numpy as np

CAFFE_ROOT = '/home/yi/code/tools/caffe/python'
sys.path.append(CAFFE_ROOT)
import caffe

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)


def extract_feature(image_list, feature_dir, oversample=True):
    """
    Extract image features from an image list
    """
    if not path.exists(feature_dir):
        os.mkdir(feature_dir)
    caffe_root = '/home/yi/code/tools/caffe'
    pretrained_model = path.join(caffe_root, 'models/vggnet/VGG_ILSVRC_16_layers.caffemodel')
    model_def = path.join(caffe_root, 'models/vggnet/VGG_ILSVRC_16_layers_deploy.prototxt')
    mean_file = path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    # image_list = './coco.list'
    # feature_dir = './vgg_fc7_4096'
    batch_size = 10000
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(model_def, pretrained_model, caffe.TEST)
    # configure pre-processing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))  # mean pixel
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_raw_scale('data', 255)
    # the reference model has channels in BGR order instead of RGB
    transformer.set_channel_swap('data', (2, 1, 0))
    image_dims = [256, 256]
    crop_dims = [224, 224]
    image_files = io.open(image_list, 'rb').readlines()
    logging.info('Total number of images %d', len(image_files))
    batch_num = 0
    sample_num = 0
    image_feature = {}
    for num_file in xrange(len(image_files)):
        file_name = image_files[num_file].strip()
        try:
            # images = [caffe.io.load_image(image_file.strip()) for image_file in [file_name]]
            image = caffe.io.load_image(file_name.strip())
        except IOError:
            continue
        except  ValueError:
            continue
        if oversample:
            height, width = image.shape[0], image.shape[1]
            if height > width:
                image_dims[0] = int(round(height * 256.0 / width))
                image_dims[1] = 256
            else:
                image_dims[0] = 256
                image_dims[1] = int(round(width * 256.0 / height))
            # print image_dims
            # input_ = np.zeros((len(images), image_dims[0], image_dims[1], 3), dtype=np.float32)
            # for i, in_ in enumerate(images):
            #    input_[i] = caffe.io.resize_image(in_, image_dims)
            input_ = np.zeros((1, image_dims[0], image_dims[1], 3), dtype=np.float32)
            input_[0] = caffe.io.resize_image(image, image_dims)
            input_ = caffe.io.oversample(input_, crop_dims)
        else:
            # Resize the image
            # input_ = np.zeros((len(images), crop_dims[0], crop_dims[1], 3), dtype=np.float32)
            # for i, in_ in enumerate(images):
            #    input_[i] = caffe.io.resize_image(in_, crop_dims)
            input_ = np.zeros((1, crop_dims[0], crop_dims[1], 3), dtype=np.float32)
            input_[0] = caffe.io.resize_image(image, crop_dims)
        # Extract features
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]], dtype=np.float32)
        for i, in_ in enumerate(input_):
            caffe_in[i] = transformer.preprocess('data', in_)
        out = net.forward_all(**{'data': caffe_in, 'blobs': ['fc7']})
        predictions = out['fc7']
        if oversample:
            # For oversampling, average predictions across crops
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = np.squeeze(predictions.mean(1))
        else:
            predictions = np.squeeze(predictions[0])

        key = path.splitext(path.basename(file_name))[0]
        # This is a special line for extracting image id from microsoft coco file name
        key = int(key.split('_')[-1])
        image_feature[key] = predictions
        logging.info('%d/%d, %s, %s', num_file, len(image_files), file_name, key)
        sample_num += 1
        if sample_num == batch_size:
            handle = io.open(path.join(feature_dir, 'batch_' + str(batch_num)), 'wb')
            cPickle.dump(image_feature, handle, protocol=cPickle.HIGHEST_PROTOCOL)
            logging.info('Finish batch %d', batch_num)
            batch_num += 1
            sample_num = 0
            image_feature = {}
    if sample_num > 0:
        handle = io.open(path.join(feature_dir, 'batch_' + str(batch_num)), 'wb')
        cPickle.dump(image_feature, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        logging.info('Finish batch %d', batch_num)
    logging.info('Done: make image feature batch')


def load_feature(feature_dir):
    """
    Load image feature
    """
    file_list = os.listdir(feature_dir)
    file_list = [path.join(feature_dir, f) for f in file_list]
    features = {}
    for file_name in file_list:
        logging.info('Load feature file %s', file_name)
        feature = cPickle.load(io.open(file_name, 'rb'))
        features.update(feature)
    return features


if __name__ == '__main__':
    extract_feature('debug.list', './', oversample=True)
