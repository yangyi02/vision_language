#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gaohaoyuan
# @Date:   2015-01-12 15:55:51
# @Last Modified by:   gaohaoyuan
# @Last Modified time: 2015-01-27 15:42:59
import json
import sys
import os
import logging
from optparse import OptionParser
from struct import unpack
import numpy as np
import threading
import pickle
import scipy.io as sio

logging.basicConfig(
	format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
)
logger = logging.getLogger('GEN Image Sentences Batch')
logger.setLevel(logging.DEBUG)

def read_kv_data(fi):
    test_key=fi.read(4)
    if test_key:
        key_len=unpack('I',test_key)[0]
        key=fi.read(key_len)
        value_len=unpack('I',fi.read(4))[0]
        value_data=fi.read(value_len)
    else :
        key_len = -1
        key = value_data = []
        value_len = -1
    return key_len, key,value_len, value_data

def decode_float_data(data):
    value = np.frombuffer(data,dtype=np.float32)
    return value

def decode_pickle_data(data):
    value = pickle.loads(data)
    return value

def get_sentence_pickle(feature_path):
	pass

def load_vgg_feature(feature_path,json_path):
	data = sio.loadmat(feature_path)['feats']
	info = json.load(open(json_path))
	dct = {}
	for line in info['images']:
		dct[str(line['cocoid'])] = data[:,line['imgid']]

	logger.info("input data length:%d", len(dct))
	return dct



def load_kv_feature_to_memory(data_dir):
	fileLists = os.listdir(data_dir)
	fileLists = [os.path.join(data_dir,i) for i in fileLists]
	feature_number = 0
	dic = {}
	for file_name in fileLists:
		fi = open(file_name,'rb')
		sys.stderr.write("Begin to load file %s\n" % file_name)
		while 1:
			key_len, key, value_len, value =  read_kv_data(fi)
			if type(key) == str:
				key=key.split('/')[-1]
			if key_len == -1:
				break
				print key
			dic[key] = decode_float_data(value)
			feature_number += 1

	logger.info("Loading feature finished, Number:%d" % feature_number)

	return dic

def load_feature_to_memory(data_dir):
	fileLists = os.listdir(data_dir)
	fileLists = [os.path.join(data_dir,i) for i in fileLists]
	feature_number = 0
	dic = {}
	for file_name in fileLists:
		fi = open(file_name,'rb')
		sys.stderr.write("Begin to load file %s\n" % file_name)
		data = pickle.load(fi)
		for i in xrange(len(data['image_id'])):
			dic[data['image_id'][i]] = data['data'][i]
			feature_number += 1

	logger.info("Loading feature finished, Number:%d" % feature_number)

	return dic

class gen_data_thread(threading.Thread):
	def __init__(self, threadID, batch_data, dct, output_file_name):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.batch_data = batch_data
		self.dct = dct
		self.output_file_name = output_file_name

	def run(self):
		fpjh.gen_proto_file( self.batch_data, self.dct, 4096, fpjh.OOV_POLICY_USE, self.output_file_name)

def main(parser,output_dir):
	# get the dictionary
	#dct = gen_dictionary(parser.dictionary_path)

	# load dct generated by Junhua
	dct = pickle.load(open(parser.dictionary_path))

	# check sign
	if (dct.get('$$S$$',-1) == -1):
		dct['$$S$$'] = len(dct)
	if (dct.get('$$E$$',-1) == -1):
		dct['$$E$$'] = len(dct)


	# load feature
	if (parser.use_vgg == 1):
		feature = load_vgg_feature(parser.vgg_feature,parser.vgg_json)
	else:
		feature = load_feature_to_memory(parser.feature_path)

	# gen batch data
	batch_data = []
	batch_name = 0
	total_number = parser.size
	output_path = output_dir
	_type = parser.type
	data = open(parser.input_list).readlines()
	fo = open('./test.file.temp','w')
	for line in data:
		line = line.strip().split("\t")
		if len(line) != 2:
			logger.info("error line %s" % line[0])

		key = line[0].strip()
		print >> fo, line[1]
		word = line[1].strip().split('  ')
		one_batch = {}
		if feature.has_key(key)== False:
			continue
		one_batch['feature'] = feature[key]
		one_batch['id'] = key
		one_batch['sentences'] = ['$$S$$'] + word + ['$$E$$']
		coding_sentences = []
		for word in one_batch['sentences']:
			id_c = dct.get(word,0)
			coding_sentences.append(id_c)

		one_batch['coding_sentences'] = coding_sentences

		batch_data.append(one_batch)

		if ( len(batch_data) >= total_number ):
			output_file_name = os.path.join(output_path, _type + "_" + str(batch_name))
			pickle.dump(batch_data,open(output_file_name,'w'))
			batch_name += 1
			batch_data = []
			sys.stderr.write("finish batch %d\n" % batch_name)

	if len(batch_data) != total_number:
		output_file_name = os.path.join(output_path, _type + "_" + str(batch_name))
		pickle.dump(batch_data,open(output_file_name,'w'))
		sys.stderr.write("finish batch %d\n" % batch_name)






if __name__ == '__main__':
	parser = OptionParser()
	usage = "useage: python %s [options] output_dir" % sys.argv[0]
	parser.add_option("-o", "--output_dir", dest="output_dir", default='./label_output', type = "string", help ="output_dir")
	parser.add_option("-f","--feature_path", dest="feature_path",default = "./data/yinan_feature/", type = "string", help="the image_feature dir")
	parser.add_option("-s","--size", dest="size", default = 3072, type = "int", help="the image number in each batch")
	parser.add_option("-d", "--dict", dest="dictionary_path",default ='./label_chinese/train.dicts', type ="string", help="input the dictionary path")
	parser.add_option("-i", dest="input_list", default = './label_chinese/1130/join.train.1130.list.seg',help="input list")
	parser.add_option('-t', '--type', dest='type', default='train', help='define the name of output')
	parser.add_option('--StartSign', dest='start_sign', default=1, type= "int", help='define the name of output')
	parser.add_option('--use_vgg', dest='use_vgg', type = "int", default= 0, help='set to use vgg feature')
	parser.add_option("--vf", dest="vgg_feature", default = './data/vgg_feature/coco/vgg_feats.mat',help="vgg_feature input")
	parser.add_option("--vj", dest="vgg_json", default = './data/vgg_feature/coco/dataset.json',help="vgg_json input")
	(options, args) = parser.parse_args()
	output_dir = options.output_dir
	print output_dir
	main(options,output_dir)