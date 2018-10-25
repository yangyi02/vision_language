# -*- coding: utf-8 -*-

from math import sqrt
import os
import sys

from trainer.recurrent_units import LstmRecurrentUnit
model_type('recurrent_nn')

# data setting
gen_list = get_config_arg('gen_list', str, './gen.list')
result_file = get_config_arg('result_file', str, './result.txt')

# dictionary setting
dict_file = get_config_arg('dict_file', str, './dict.txt')
dict_pkl = get_config_arg('dict_pkl', str, './dict.pkl')

# image feature setting
img_feat_list = get_config_arg('img_feat_list', str, './img_feat.list')

# feature dimension setting
img_feat_dim = get_config_arg('img_feat_dim', int, 4096)
word_embedding_dim = 512
hidden_dim = 512
multimodal_dim = 1024
dict_dim = len(open(dict_file).readlines())
start_index = dict_dim-2
end_index = dict_dim-1

# hyperparameter setting
Settings(
    batch_size = 8, # this must equal to trainer_count
    learning_rate = 0,
)

# data provider setting
TestData(
    PyData(
        files = gen_list,
        load_data_module = 'join_test',
        load_data_object = 'processData',
        load_data_args = ' '.join([dict_pkl, img_feat_list, str(img_feat_dim), '1.0'])
    )
)

##### network #####
Inputs('question_id', 'img_feat', 'question')
Outputs('predict_word')

# data layers
DataLayer(name = 'question_id', size = 1)
DataLayer(name = 'img_feat', size = img_feat_dim)
DataLayer(name = 'question', size = dict_dim)

# rnn1
RecurrentLayerGroupBegin('rnn1' + '_layer_group',
    in_links = [],
    out_links = ['predict_word'],
    seq_reversed = False,
    generator = Generator(
        max_num_frames = 20,
        beam_size = 5,
        num_results_per_sample = 1,
    ),
)

predict_word_memory = Memory(name = 'predict_word',
    size = dict_dim,
    boot_with_const_id = start_index,
)

MixedLayer(name = 'predict_word_embedding',
    size = word_embedding_dim,
    bias = False,
    inputs = TableProjection(predict_word_memory,
        parameter_name = 'word_embedding',
    ),
)

# hidden1
MixedLayer(name = 'hidden1',
    size = hidden_dim,
    active_type = 'stanh',
    bias = Bias(parameter_name = '_hidden1.wbias'),
    inputs = [FullMatrixProjection('predict_word_embedding',
        parameter_name = '_hidden1.w0')],
)

LstmRecurrentUnit(name = 'rnn1',
    size = hidden_dim/4,
    active_type = 'relu',
    state_active_type = 'linear',
    gate_active_type = 'sigmoid',
    inputs = [IdentityProjection('hidden1')],
)

img_feat_memory = Memory(name = 'img_feat_memory',
    size = img_feat_dim,
    boot_layer = 'img_feat',
    is_sequence = False,
)

MixedLayer(name = 'img_feat_memory',
    size = img_feat_dim,
    bias = False,
    inputs = IdentityProjection(img_feat_memory),
)

# hidden2
MixedLayer(name = 'hidden2',
    size = multimodal_dim,
    active_type = 'stanh',
    bias = Bias(parameter_name = '_hidden2.wbias'),
    inputs = [FullMatrixProjection('predict_word_embedding', parameter_name = '_hidden2.w0'),
        FullMatrixProjection(img_feat_memory, parameter_name = '_hidden2.w1'),
        FullMatrixProjection('rnn1', parameter_name = '_hidden2.w2'),
    ],
    # drop_rate = 0.5,
)

# hidden3
#Layer(
#    name = 'hidden3',
#    type = 'mixed',
#    size = word_embedding_dim,
#    active_type = 'stanh',
#    inputs = FullMatrixProjection(
#        'hidden2',
#        initial_std = sqrt(1. / multimodal_dim)),
#)

# output
Layer(name = 'output',
    type = 'fc',
    size = dict_dim,
    active_type = 'softmax',
    bias = Bias(parameter_name = '_output.wbias'),
    inputs = [Input('hidden2', parameter_name = '_output.w0')],
    #inputs = TransposedFullMatrixProjection(
    #    'hidden3',
    #    parameter_name = 'wordvecs'),
)

Layer(
    name = 'predict_word',
    type = 'maxid',
    inputs = 'output',
)

Layer(
    name = 'eos_check',
    type = 'eos_id',
    eos_id = end_index,
    inputs = ['predict_word'],
)

RecurrentLayerGroupEnd('rnn1' + '_layer_group')

# Write question and answer pairs to file
Evaluator(
    name = 'caption_printer',
    type = 'seq_text_printer',
    dict_file = dict_file,
    result_file = result_file,
    #delimited = False,
    inputs = ['question_id', 'question', 'predict_word'],
)

