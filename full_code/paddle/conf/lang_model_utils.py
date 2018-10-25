from trainer.config_parser import *
from trainer.recurrent_units import *

def PName(prefix, name):
    return "%s_%s" % (prefix, name)

def PredictionModel(para_prefix, input_layer,
                    conf, name_prefix=''):
    Layer(
       name = name_prefix + 'output_word_embedding',
       type = 'mixed',
       size = conf.word_embedding_dim,
       active_type = "stanh",
       bias = Bias(parameter_name = PName(para_prefix, "output_word_embedding.bias")),
       inputs = FullMatrixProjection(input_layer,
                                     parameter_name = PName(para_prefix, "output_word_embedding"),
                                     initial_std = math.sqrt(1./conf.hidden_dim))
    )

    Layer(name = name_prefix + "softmax_layer",
          type = "mixed",
          size = conf.dict_dim,
          active_type = "softmax",
          bias = Bias(parameter_name = PName(para_prefix, "softmax_layer.bias"),
                      initial_mean = 0,
                      initial_std = 0),
          inputs = TransposedFullMatrixProjection(name_prefix + "output_word_embedding",
                                                  parameter_name="%s_wordvecs" % para_prefix)
     )
    if conf.generating:
        Layer(name = name_prefix + "predict",
              type = "maxid",
              inputs = name_prefix + "softmax_layer")


def WordEmbeddingLayer(para_prefix, name, input_layer, conf):
    Layer(
        name = name,
        type = "mixed",
        size = conf.word_embedding_dim,
        bias = False,
        inputs = TableProjection(
            input_layer,
            parameter_name="%s_wordvecs" % para_prefix),
    )

def ExpandInput(name, input_layer, input_dim):
    input_memory = Memory(name = name,
                             size = input_dim,
                             boot_layer = input_layer)
    Layer(name = name,
          type = "mixed",
          active_type = "",
          bias = False,
          inputs = IdentityProjection(input_memory)
    )
    return input_memory


def ImageAndRnnGate(conf, encoding_prefix,
                    img_layer_name = 'img_expand'):
    Layer(name = encoding_prefix +"rnn_gate",
          type = "mixed",
          active_type = "sigmoid",
          size = 1,
          bias = Bias(parameter_name = "rnn_gate.b",
                  initial_std = 0.0,
                  initial_mean = -2.0),
          inputs = [FullMatrixProjection(encoding_prefix +"rnn",
                                     parameter_name = "rnn_gate_proj")]
          )
    Layer(name = encoding_prefix +"image_gate",
          type = "mixed",
          size = 1,
          bias = Bias(parameter_name = "image_gate.b",
                  initial_std = 0.0,
                  initial_mean = -2.0),
          active_type = "sigmoid",
          inputs = [FullMatrixProjection(encoding_prefix +"rnn",
                                         parameter_name = "image_gate_proj")]
          )

    Layer(name = encoding_prefix +"rnn_gate_expanded",
          type = "featmap_expand",
          num_filters = conf.rnn_size,
          inputs = [FullMatrixProjection(encoding_prefix +"rnn_gate")]
          )

    Layer(name = encoding_prefix + "image_gate_expanded",
          type = "featmap_expand",
          num_filters = conf.img_feat_dim,
          inputs = [FullMatrixProjection(encoding_prefix +"image_gate")]
          )

    Layer(name = encoding_prefix +"gated_rnn",
          type = "mixed",
          bias = False,
          size = conf.rnn_size,
          inputs = [DotMulOperator([encoding_prefix +"rnn_gate_expanded",
                    encoding_prefix + "rnn"])]
        )

    Layer(name = encoding_prefix +"gated_img",
          type = "mixed",
          bias = False,
          size = conf.img_feat_dim,
          inputs = [DotMulOperator([encoding_prefix + "image_gate_expanded",
                    img_layer_name])]
        )

def LangModel(para_prefix, img_feat_layer, input_layer,
              conf, drop_rate, initial_state_layer = None, name_prefix = '',
              with_gate = False):
    if not conf.generating:
        WordEmbeddingLayer(para_prefix, name_prefix + "word_embedding", input_layer, conf)

    RecurrentLayerGroupBegin(name_prefix + "rnn_layer_group",
                             in_links=[] if conf.generating else [name_prefix + 'word_embedding'],
                             out_links=[name_prefix + 'predict', name_prefix + 'rnn'] if conf.generating else [name_prefix + 'rnn'],
                             generator = Generator(max_num_frames=50,
                                                   beam_size = 5,
                                                   eos_layer_name = name_prefix + "eos_check",
                                                   num_results_per_sample = 1) if conf.generating else None,
                             seq_reversed=False)
    if conf.generating: # generated word
        predict_word_memory = Memory(name = name_prefix + "predict",
                                     size = conf.dict_dim,
                                     boot_with_const_id = conf.dict_eos)
        WordEmbeddingLayer(para_prefix, name_prefix + "word_embedding", predict_word_memory, conf)

    # RNN gate transformations.
    input_layer_name = name_prefix + "rnn_transformed_input"
    Layer(
          name = input_layer_name,
          type = "mixed",
          size = conf.rnn_size * 3,
          active_type = "",
          bias = False,
          inputs = [FullMatrixProjection(name_prefix + "word_embedding",
                                     parameter_name = PName(para_prefix, "rnn_transform_input"))],
    )

    if initial_state_layer:
        out_memory = Memory(name = name_prefix + "rnn",
                        size = conf.rnn_size,
                        boot_layer = initial_state_layer)
    else:
        out_memory = Memory(name = name_prefix + "rnn",
                        size = conf.rnn_size)
    # recurrent unit.
    GatedRecurrentUnit(
        name = name_prefix + "rnn",
        size = conf.rnn_size,
        active_type = "tanh",
        gate_active_type = "sigmoid",
        inputs = input_layer_name, #transform outside
        out_memory = out_memory,
        para_prefix = para_prefix,
        )


    if conf.generating:
        img_feat_mem = ExpandInput(name = name_prefix + "img_feat",
                               input_layer = img_feat_layer,
                               input_dim = conf.img_feat_dim)
        if with_gate:
            ImageAndRnnGate(conf, name_prefix, img_feat_mem)
            Layer(
              name = name_prefix + "word_rnn_mix",
              type = "mixed",
              size = conf.hidden_dim,
              active_type = "linear",
              bias = False,
              inputs = [
                  FullMatrixProjection(name_prefix + "word_embedding", parameter_name = '%s_we_proj.w' % para_prefix),
                  FullMatrixProjection(name_prefix + "gated_img", parameter_name = '%s_img_proj.w' % para_prefix),
                  FullMatrixProjection(name_prefix + "rnn", parameter_name = '%s_lstm_proj.w' % para_prefix),
              ],
              drop_rate = drop_rate,
            )
        else:
            Layer(
              name = name_prefix + "word_rnn_mix",
              type = "mixed",
              size = conf.hidden_dim,
              active_type = "linear",
              bias = False,
              inputs = [
                  FullMatrixProjection(name_prefix + "word_embedding", parameter_name = '%s_we_proj.w' % para_prefix),
                  FullMatrixProjection(img_feat_mem, parameter_name = '%s_img_proj.w' % para_prefix),
                  FullMatrixProjection(name_prefix + "rnn", parameter_name = '%s_lstm_proj.w' % para_prefix),
              ],
              drop_rate = drop_rate,
            )
        PredictionModel(para_prefix = para_prefix,
                        input_layer = name_prefix + 'word_rnn_mix',
                        conf = conf, name_prefix = name_prefix)

        Layer(name = name_prefix + "eos_check",
              type = "eos_id",
              eos_id = conf.dict_eos,
              inputs = [Input(name_prefix + "predict")])


    RecurrentLayerGroupEnd(name_prefix + "rnn_layer_group")


def LangEncodingModel(conf, encoding_prefix, drop_rate,
                      para_prefix,
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
        rnn_input = encoding_prefix +"gated_rnn"
    else:
        rnn_input = encoding_prefix + "rnn"

    Layer(
        name = encoding_prefix + "word_rnn_mix",
        type = "mixed",
        size = conf.hidden_dim,
        active_type = "stanh",
        bias = False,
        inputs = [
            FullMatrixProjection(encoding_prefix +"word_embedding", parameter_name = '%s_we_proj.w' % para_prefix),
            FullMatrixProjection(encoding_prefix +"img_expand", parameter_name = '%s_img_proj.w' % para_prefix),
            FullMatrixProjection(rnn_input, parameter_name = '%s_lstm_proj.w' % para_prefix),
        ],
          drop_rate = drop_rate,
    )
    PredictionModel(para_prefix = para_prefix,
                    input_layer = encoding_prefix + 'word_rnn_mix',
                    conf = conf,
                    name_prefix = encoding_prefix)

class LangModelConf():
    def __init__(self):
        self.img_feat_dim = 4096
