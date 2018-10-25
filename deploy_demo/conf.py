# -*- coding: utf-8 -*-

from math import sqrt
import os
import sys

from paddle.trainer.recurrent_units import LstmRecurrentUnit
model_type('recurrent_nn')

# data setting
file_list = get_config_arg('file_list', str, './file.list')
result_file = get_config_arg('result_file', str, './result.txt')

# dictionary setting
dict_file = get_config_arg('dict_file', str, './dict.txt')
dict_pkl = get_config_arg('dict_pkl', str, './dict.pkl')

# feature dimension setting
image_feat_dim = get_config_arg('image_feat_dim', int, 4096)
word_embedding_dim = 512
hidden_dim = 512
multimodal_dim = 1024
dict_dim = len(open(dict_file).readlines())
start_index = dict_dim-2
end_index = dict_dim-1

# batch_size must equal to trainer_count
batch_size = get_config_arg('batch_size', int, 1)

# hyperparameter setting
Settings(
    batch_size = batch_size, # This must be equal to trainer_count
    learning_rate = 0,
)

# data provider setting
TestData(
    PyData(
        files = file_list,
        load_data_module = 'data_provider',
        load_data_object = 'processData',
        load_data_args = ' '.join([dict_pkl, str(image_feat_dim), '1.0'])
    )
)

##### network #####
Inputs('question_id', 'image_feat', 'question')
Outputs('predict_word')

# data layers
DataLayer(name = 'question_id', size = 1)
DataLayer(name = 'image_feat', size = image_feat_dim)
DataLayer(name = 'question', size = dict_dim)

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

predict_word_memory = Memory(
    name = 'predict_word',
    size = dict_dim,
    boot_with_const_id = start_index,
)

MixedLayer(
    name = 'predict_word_embedding',
    size = word_embedding_dim,
    bias = False,
    inputs = TableProjection(predict_word_memory,
        parameter_name = 'word_embedding',
        initial_std = 0,
    ),
)

# hidden1
MixedLayer(
    name = 'hidden1',
    size = hidden_dim,
    active_type = 'stanh',
    bias = Bias(parameter_name = '_hidden1.wbias'),
    inputs = [
        FullMatrixProjection('predict_word_embedding',
            parameter_name = '_hidden1.w0'),
    ]
)

LstmRecurrentUnit(
    name = 'rnn1',
    size = hidden_dim/4,
    active_type = 'relu',
    state_active_type = 'linear',
    gate_active_type = 'sigmoid',
    inputs = [IdentityProjection('hidden1')],
    para_prefix = None,
    error_clipping_threshold = 0,
)

image_feat_memory = Memory(
    name = 'image_feat_memory',
    size = image_feat_dim,
    boot_layer = 'image_feat',
    is_sequence = False,
)

MixedLayer(
    name = 'image_feat_memory',
    size = image_feat_dim,
    bias = False,
    inputs = IdentityProjection(image_feat_memory),
)

# hidden2
MixedLayer(
    name = 'hidden2',
    size = multimodal_dim,
    active_type = 'stanh',
    bias = Bias(parameter_name = '_hidden2.wbias'),
    inputs = [
        FullMatrixProjection('hidden1', parameter_name = '_hidden2.w0'),
        FullMatrixProjection(image_feat_memory, parameter_name = '_hidden2.w1'),
        FullMatrixProjection('rnn1', parameter_name = '_hidden2.w2'),
    ],
    drop_rate = 0.5
)

# output
MixedLayer(
    name = 'output',
    size = dict_dim,
    active_type = 'softmax',
    bias = Bias(parameter_name = '_output.wbias'),
    inputs = FullMatrixProjection('hidden2', parameter_name = '_output.w0'),
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

