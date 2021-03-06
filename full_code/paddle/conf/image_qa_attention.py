# -*- coding: utf-8 -*-

from math import sqrt
import os
import sys

from trainer.recurrent_units import LstmRecurrentUnit
model_type('recurrent_nn')

# data setting
train_list = get_config_arg('train_list', str, './train.list')
test_list = get_config_arg('test_list', str, './val.list')

# dictionary setting
dict_file = get_config_arg('dict_file', str, './dict.txt')
dict_pkl = get_config_arg('dict_pkl', str, './dict.pkl')

# image feature setting
image_feat_list = get_config_arg('image_feat_list', str, './image_feat.list')

# feature dimension setting
image_feat_dim = get_config_arg('image_feat_dim', int, 4096)
word_embedding_dim = 512
hidden_dim = 512
multimodal_dim = 1024
dict_dim = len(open(dict_file).readlines())

# hyperparameter setting
default_decay_rate(1e-4)
default_initial_mean(0)
default_initial_std(1.0 / sqrt(hidden_dim))

Settings(
    batch_size = 16,
    algorithm = 'sgd',
    learning_rate = 5e-3,
    learning_rate_decay_a = 5e-7,
    learning_rate_decay_b = 0.5,
)

#Settings(
#    batch_size = 16,
#    algorithm = 'sgd',
#    learning_method = 'adadelta',
#    ada_epsilon = 0.01,
#    ada_rou = 0.99,
#    learning_rate = 5e-3,
#    learning_rate_decay_a = 5e-7,
#    learning_rate_decay_b = 0.5,
#)

# data provider setting
TrainData(
    PyData(
        files = train_list,
        load_data_module = 'join_train',
        load_data_object = 'processData',
        load_data_args = ' '.join([dict_pkl, image_feat_list, str(image_feat_dim), '1.0']),
        async_load_data = True,
    )
)
TestData(
    PyData(
        files = test_list,
        load_data_module = 'join_val',
        load_data_object = 'processData',
        load_data_args = ' '.join([dict_pkl, image_feat_list, str(image_feat_dim), '1.0']),
    )
)

##### network #####
Inputs('image_feat', 'question', 'word', 'next_word')
Outputs('cost')

# data layers
DataLayer(name = 'image_feat', size = image_feat_dim)
DataLayer(name = 'question', size = dict_dim)
DataLayer(name = 'word', size = dict_dim)
DataLayer(name = 'next_word', size = dict_dim)

# question embedding input: question_embedding
MixedLayer(name = 'question_embedding',
    size = word_embedding_dim,
    bias = False,
    inputs = TableProjection('question',
        parameter_name = 'word_embedding',
        initial_std = 1.0 / sqrt(word_embedding_dim),
    ),
)

# answer embedding input: word_embedding
MixedLayer(name = 'word_embedding',
    size = word_embedding_dim,
    bias = False,
    inputs = TableProjection('word',
        parameter_name = 'word_embedding',
        initial_std = 1.0 / sqrt(word_embedding_dim),
    ),
)

# question hidden input
MixedLayer(name = 'question_input',
    size = hidden_dim,
    active_type = 'stanh',
    inputs = FullMatrixProjection('question_embedding',
        initial_std = 1.0 / sqrt(hidden_dim)),
)

# question hidden input: encoder
RecurrentLayerGroupBegin('encoder' + '_layer_group',
    in_links = ['question_input'],
    out_links = ['encoder'],
    seq_reversed = False,
)

LstmRecurrentUnit(name = 'encoder',
    size = hidden_dim/4,
    active_type = 'relu',
    state_active_type = 'linear',
    gate_active_type = 'sigmoid',
    inputs = [IdentityProjection('question_input')],
)

RecurrentLayerGroupEnd('encoder' + '_layer_group')

# get last of encoder
Layer(name = 'encoder_last',
    type = 'seqlastins',
    active_type = '',
    bias = False,
    inputs = [Input('encoder')],
)

# compute question attention on image feature
Layer(name = 'question_filter',
    type = 'fc',
    size = image_feat_dim,
    bias = True,
    inputs = [FullMatrixProjection('encoder_last')],
)

Layer(name = 'image_atten',
    type = 'mixed',
    size = 1,
    bias = False,
    active_type='softmax',
    inputs = ConvOperator(input_layer_names=['image_feat', 'question_filter'],
        num_filters=1,
        conv_conf=Conv(filter_size=1,
            channels = image_feat_dim,
            padding = 0,
            stride = 1,
            groups = 1)),
)

Layer(name = 'image_atten_expand',
    type = 'featmap_expand',
    num_filters = image_feat_dim,
    inputs = [Input('image_atten')],
)

MixedLayer(name = 'image_feat_after_atten',
    size = image_feat_dim,
    bias = False,
    inputs = DotMulOperator(input_layer_names=['image_atten_expand', 'image_feat'],
        scale = 1.0),
)

# hidden1
MixedLayer(name = 'hidden1',
    size = hidden_dim, # This must be 4 times recurrent unit size
    active_type = 'stanh',
    inputs = FullMatrixProjection('word_embedding',
        initial_std = 1.0 / sqrt(hidden_dim)),
)

# rnn1
RecurrentLayerGroupBegin('rnn1' + '_layer_group',
    in_links = ['hidden1'],
    out_links = ['rnn1'],
    seq_reversed = False,
)

LstmRecurrentUnit(name = 'rnn1',
    size = hidden_dim/4,
    active_type = 'relu',
    state_active_type = 'linear',
    gate_active_type = 'sigmoid',
    inputs = [IdentityProjection('hidden1')],
)

RecurrentLayerGroupEnd('rnn1' + '_layer_group')

# image feature input: image_expand
Layer(name = 'image_expand',
    type = 'expand',
    bias = False,
    inputs = ['image_feat_after_atten', 'word'],
)

# question enocder input: question_expand
Layer(name = 'question_expand',
    type = 'expand',
    bias = False,
    inputs = ['encoder_last', 'word'],
)

# hidden2
MixedLayer(name = 'hidden2',
    size = multimodal_dim,
    active_type = 'stanh',
    inputs = [FullMatrixProjection('question_expand'),
        FullMatrixProjection('word_embedding'),
        FullMatrixProjection('image_expand',
                            learning_rate = 0.01,
                            momentum = 0.9,
                            decay_rate = 0.05,
                            initial_std = 0.01),
        FullMatrixProjection('rnn1'),
    ],
    # drop_rate = 0.5,
)

# hidden3
#MixedLayer(
#    name = 'hidden3',
#    size = word_embedding_dim,
#    active_type = 'stanh',
#    inputs = FullMatrixProjection('hidden2',
#        initial_std = 1.0 / sqrt(multimodal_dim)),
#)

# output
Layer(name = 'output',
    type = 'fc',
    size = dict_dim,
    active_type = 'softmax',
    inputs = [Input('hidden2',
        initial_std = 1.0 / sqrt(word_embedding_dim),
        learning_rate = 0.01)],
)

# cost
Layer(name = 'cost',
    type = 'multi-class-cross-entropy',
    inputs = ['output', 'next_word'],
)

Evaluator(name = 'classification_error',
    type = 'classification_error',
    inputs = ['output', 'next_word'],
)
