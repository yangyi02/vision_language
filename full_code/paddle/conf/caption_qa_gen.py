#edit-mode: -*- python -*-
# created by Wang Jiang
from math import sqrt
import os
import sys
from trainer.recurrent_units import *


model_type('recurrent_nn')

generating = get_config_arg('generating', int, 1)

# data setting
if generating:
    gen_list = get_config_arg('gen_list', str, './gen.list')
    result_file = get_config_arg('result_file', str, './result.txt')
else:
    train_list = get_config_arg('train_list', str, './train.list')
    test_list = get_config_arg('test_list', str, './val.list')

# dictionary setting
dict_file = get_config_arg('dict_file', str,
                           './dict.txt')
dict_pkl = get_config_arg('dict_pkl', str, './dict.pkl')

# image feature setting
img_feat_list = get_config_arg('img_feat_list', str, './img_feat.list')

img_feat_dim = get_config_arg('img_feat_dim', int, 4096)

dict_dim = len(open(dict_file).readlines())
print ("dict dim: %d " % dict_dim)
dict_bos = dict_dim - 2
dict_eos = dict_dim - 1

from lang_model_utils import *

conf = LangModelConf()
conf.dict_dim = dict_dim
conf.dict_bos = dict_bos
conf.dict_eos = dict_eos
conf.generating = generating
conf.img_feat_dim = img_feat_dim
conf.word_embedding_dim = 512
conf.hidden_dim = 1024
conf.rnn_size = 128
conf.gate_size = 4

# hyperparameter setting
default_initial_mean(0)
default_initial_strategy(0) # 0 for normal, 1 for uniform
default_initial_smart(True)
default_decay_rate(1e-4)
default_num_batches_regularization(1)
default_gradient_clipping_threshold(25)

if generating:
    Settings(
        batch_size = 1,
        learning_rate = 0,
    )
else:
    Settings(
        batch_size = 16,
        algorithm = 'sgd',
        learning_method = 'adadelta',
        ada_epsilon = 0.01,
        ada_rou = 0.99,
        learning_rate = 1e-3,
        learning_rate_schedule = "constant",
        learning_rate_decay_a = 5e-7,
        learning_rate_decay_b = 0.5,
    )

#load_data_args = dict_pkl + ' 10000'

# data setting
if generating:
    TestData(
        PyData(
            files = gen_list,
            #load_data_module = 'pyJoin_generate_dialog_and_qa_synthetic',
            load_data_module = 'join_test_caption_qa',
            load_data_object = 'processData',
            load_data_args = ' '.join([dict_pkl, img_feat_list, str(img_feat_dim), '1.0']),
        )
    )
else:
    TrainData(
        PyData(
            files = train_list,
            #load_data_module = 'pyJoin_train_dialog_and_qa_synthetic',
            load_data_module = 'join_train_caption_qa',
            load_data_object = 'processData',
            load_data_args = ' '.join([dict_pkl, img_feat_list, str(img_feat_dim), '1.0']),
            async_load_data = True,
        )
    )

    TestData(
        PyData(
            files = test_list,
            #load_data_module = 'pyJoin_test_dialog_and_qa_synthetic',
            load_data_module = 'join_val_caption_qa',
            load_data_object = 'processData',
            load_data_args = ' '.join([dict_pkl, img_feat_list, str(img_feat_dim), '1.0']),
        )
    )

if conf.generating:
    Inputs('question_id', 'img_feat', 'que')
    DataLayer(name = 'question_id', size = 1)
    DataLayer(name = 'img_feat', size = conf.img_feat_dim)
    DataLayer(name = 'que', size = conf.dict_dim)
    DataLayer(name = 'last_state_gate', size = conf.gate_size)
else:
    Inputs('img_feat', 'que', 'que_next', 'ans', 'ans_next')
    DataLayer(name = 'img_feat', size = conf.img_feat_dim)
    DataLayer(name = 'que', size = conf.dict_dim)
    DataLayer(name = 'que_next', size = conf.dict_dim)
    DataLayer(name = 'ans', size = conf.dict_dim)
    DataLayer(name = 'ans_next', size = conf.dict_dim)


def LangEncodingModel(conf, encoding_prefix, drop_rate,
                      input_layer, initial_state=None,
                      with_gate = False):
    if conf.generating:
        enc_conf = copy.deepcopy(conf)
        enc_conf.generating = False
    else:
        enc_conf = conf
    LangModel(para_prefix = para_prefix,
              img_feat_layer = None,
              input_layer = input_layer,
              conf = enc_conf,
              drop_rate = drop_rate,
              initial_state_layer = initial_state,
              name_prefix = encoding_prefix
    )

    Layer(
        name = encoding_prefix + 'img_expand',
        type = 'expand',
        bias = False,
        inputs = ['img_feat', input_layer],
    )

    if with_gate:
        ImageAndRnnGate(conf, encoding_prefix, encoding_prefix + 'img_expand')
        rnn_input = encoding_prefix +"rnn"
        image_input = encoding_prefix + "gated_img"
    else:
        rnn_input = encoding_prefix + "rnn"
        image_input = encoding_prefix +"img_expand"

    Layer(
        name = encoding_prefix + "word_rnn_mix",
        type = "mixed",
        size = conf.hidden_dim,
        active_type = "linear",
        bias = False,
        inputs = [
            FullMatrixProjection(encoding_prefix +"word_embedding", parameter_name = '%s_we_proj.w' % para_prefix),
            FullMatrixProjection(image_input, parameter_name = '%s_img_proj.w' % para_prefix),
            FullMatrixProjection(rnn_input, parameter_name = '%s_lstm_proj.w' % para_prefix),
        ],
          drop_rate = drop_rate,
    )
    PredictionModel(para_prefix = para_prefix,
                    input_layer = encoding_prefix + 'word_rnn_mix',
                    conf = conf,
                    name_prefix = encoding_prefix)

def QuestionEncoding(conf, encoding_prefix, drop_rate, with_gate=False):
    LangEncodingModel(conf, encoding_prefix, drop_rate, 'que', with_gate = with_gate)
    Layer(
        name = encoding_prefix + 'last_state',
        type = 'seqlastins',
        active_type = '',
        bias = False,
        inputs = [Input(encoding_prefix + 'rnn')],
    )


para_prefix = "lang"
drop_rate = 0.5
with_gate = False #True
if conf.generating:
    encoding_prefix = 'encoding_'
    decoding_prefix = 'decoding_'

    #Outputs(decoding_prefix + "predict", decoding_prefix + "image_gate")
    Outputs(decoding_prefix + "predict")

    QuestionEncoding(conf, encoding_prefix, drop_rate, with_gate)

    LangModel(para_prefix = para_prefix,
              img_feat_layer = 'img_feat',
              input_layer = None,
              conf = conf,
              drop_rate = drop_rate,
              initial_state_layer = encoding_prefix + "last_state",
              name_prefix = 'decoding_',
              with_gate = with_gate
    )

    Evaluator(name = 'caption_printer',
              type = 'seq_text_printer',
              dict_file = dict_file,
              result_file = result_file,
              inputs = ['question_id', 'que', decoding_prefix + 'predict'])
else:
    encoding_prefix = 'encoding_'
    decoding_prefix = 'decoding_'
    encoding_res_prefix = 'encoding_res_'
    Outputs(encoding_prefix + "cost", decoding_prefix + "cost")
    # encoding part

    QuestionEncoding(conf, encoding_prefix, drop_rate, with_gate = with_gate)
    # cost
    Layer(
        name = encoding_prefix + 'cost',
        type = 'multi-class-cross-entropy',
        inputs = [encoding_prefix + 'softmax_layer', 'que_next'],
    )

    Evaluator(
        name = encoding_prefix + 'qa_classification_error',
        type = 'classification_error',
        inputs = [encoding_prefix + 'softmax_layer', 'que_next'],
    )

    # decoding part
    LangEncodingModel(conf, decoding_prefix, drop_rate, 'ans',
                      encoding_prefix + 'last_state', with_gate = with_gate)

    # cost
    Layer(
        name = decoding_prefix + 'cost',
        type = 'multi-class-cross-entropy',
        inputs = [decoding_prefix + 'softmax_layer', 'ans_next'],
    )

    Evaluator(
        name = decoding_prefix + 'qa_classification_error',
        type = 'classification_error',
        inputs = [decoding_prefix + 'softmax_layer', 'ans_next'],
    )


    Layer(
        name = decoding_prefix + 'last_state',
        type = 'seqlastins',
        active_type = '',
        bias = False,
        inputs = [Input(decoding_prefix + 'rnn')],
    )
