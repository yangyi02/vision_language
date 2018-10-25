# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 03:19:05 2015

@author: yangyi05
"""

import sys
import io
import logging
import cPickle

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logging.getLogger().setLevel(logging.INFO)

def remove_id(in_text):
    """
    Remove image id and question id before building dictionary
    """
    task = in_text[0].strip()
    if task == 'image caption':
        out_text = []
        for line in in_text[1:]:
            line = line.split('\t')
            line = [line[1]]
            out_text.append('\t'.join(line))
    elif task == 'image qa':
        out_text = []
        for line in in_text[1:]:
            line = line.split('\t')
            line = line[2:]
            out_text.append('\t'.join(line))
    elif task == 'qa':
        out_text = []
        for line in in_text[1:]:
            line = line.split('\t')
            line = line[1:]
            out_text.append('\t'.join(line))
    else:
        logging.fatal('Unrecognized task: %s', task)
        sys.exit(1)
    return out_text

def compute_frequency(data, word_freq_file):
    """
    Compute word frequency from string data
    """
    text = remove_id(data)
    word_freq_dict = {}
    for line in text:
        line = line.strip().split('\t')
        for sentence in line:
            words = sentence.split(' ')
            for word in words:
                if word in word_freq_dict:
                    word_freq_dict[word] += 1
                else:
                    word_freq_dict[word] = 1

    output_pair = sorted(word_freq_dict.items(), key=lambda x: x[1])
    handle = io.open(word_freq_file, 'wb')
    for item in output_pair:
        handle.write('%s\t%d\n' % (item[0], item[1]))

def make_dictionary(word_freq_file, dict_freq_thresh, dict_pkl, dict_file):
    """
    Build dictionary from word frequency file
    Will trim the dictionary by a threshold
    """
    word_freq = io.open(word_freq_file, 'rb').readlines()
    word_dict, cnt = {}, 0
    for line in word_freq:
        items = line.strip().split('\t')
        word = items[0]
        freq = int(items[1])
        if freq >= dict_freq_thresh:
            word_dict[word] = cnt
    # Rebuild dictionary with only high frequency keys
    word_dict.pop('', None)
    final_dict = {}
    cnt = 1
    for item in word_dict.items():
        final_dict[item[0]] = cnt
        cnt += 1
    # Add special keys
    final_dict['#OOV#'] = 0
    final_dict['$$S$$'] = len(final_dict)
    final_dict['$$E$$'] = len(final_dict)
    cPickle.dump(final_dict, io.open(dict_pkl, 'wb'))
    save_to_txt(final_dict, dict_file)

def save_to_txt(word_dict, dict_file):
    """
    Save the dictionary to txt file
    """
    output_pair = sorted(word_dict.items(), key=lambda x: x[1])
    handle = io.open(dict_file, 'wb')
    for item in output_pair:
        handle.write('%s\n' % item[0])

def merge_frequency(word_freq_files, out_word_freq_file):
    """
    Merge two word frequency files into one
    """
    word_freq_dict = {}
    for word_freq_file in word_freq_files:
        word_freq = io.open(word_freq_file, 'rb').readlines()
        for line in word_freq:
            items = line.strip().split('\t')
            word = items[0]
            if word in word_freq_dict:
                word_freq_dict[word] += int(items[1])
            else:
                word_freq_dict[word] = int(items[1])

    output_pair = sorted(word_freq_dict.items(), key=lambda x: x[1])
    handle = io.open(out_word_freq_file, 'wb')
    for item in output_pair:
        handle.write('%s\t%d\n' % (item[0], item[1]))

def merge_dictionary(dict_pkls, out_dict_pkl, out_dict_file):
    """
    Merge two dictionary files into one
    """
    out_word_dict = {}
    for dict_pkl in dict_pkls:
        word_dict = cPickle.load(io.open(dict_pkl, 'rb'))
        # Remove special keys
        word_dict.pop('#OOV#', None)
        word_dict.pop('$$S$$', None)
        word_dict.pop('$$E$$', None)
        out_word_dict.update(word_dict)
    # Rebuild dictionary with only high frequency keys
    word_dict.pop('', None)
    final_dict = {}
    cnt = 1
    for item in out_word_dict.items():
        final_dict[item[0]] = cnt
        cnt += 1
    # Add special keys
    final_dict['#OOV#'] = 0
    final_dict['$$S$$'] = len(final_dict)
    final_dict['$$E$$'] = len(final_dict)
    cPickle.dump(final_dict, io.open(out_dict_pkl, 'wb'))
    save_to_txt(final_dict, out_dict_file)
