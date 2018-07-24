from __future__ import print_function
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
from bigdl.keras.converter import WeightsConverter
from bigdl.optim.optimizer import Adam
from zoo.pipeline.api.keras.engine.topology import *
import numpy as np
import keras.backend as KK
from keras.engine.training import _standardize_input_data
import keras.layers as klayers

from utils import *
import inputs
import metrics

np.random.seed(1330)

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)

def load_zoo_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    model_config = config['zmodel']['setting']
    model_config.update(config['inputs']['share'])
    sys.path.insert(0, config['zmodel']['model_path'])

    model = import_object(config['zmodel']['model_py'], model_config)
    mo = model.build()
    return mo

def load_keras2_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    model_config = config['kmodel']['setting']
    model_config.update(config['inputs']['share'])
    sys.path.insert(0, config['kmodel']['model_path'])

    model = import_object(config['kmodel']['model_py'], model_config)
    mo = model.build()
    return mo


def zloss(**kwargs):
    if isinstance(kwargs, dict) and 'batch' in kwargs: #[b, 2, 1] [b, 1]
        batch = kwargs['batch']
    def _zloss(y_true, y_pred):
        y_pred = y_pred + y_true - y_true
        margin = 1.0

        pos = y_pred.index_select(1, 0)
        neg = y_pred.index_select(1, 1)

        loss = maximum(neg - pos + margin, 0.)
        return loss
    return _zloss


def kloss(y_true, y_pred):
    margin = 1.0
    y_pos = klayers.Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = klayers.Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    loss = KK.maximum(0., margin + y_neg - y_pos)
    return KK.mean(loss)


def pair(query, doc):
    result = []
    for i in range(0, query.shape[0], 2):
        t1 = np.stack((query[i], query[i + 1]), axis=0)
        t2 = np.stack((doc[i], doc[i + 1]), axis=0)
        c = np.concatenate([t1, t2], axis=1)
        result.append(c)
    return result


def preprocess(input_data):
    result = []
    for x in input_data:
        t = pair(x[0], x[1])
        result.append(t)
    result = np.concatenate(result, axis=0)
    return result


def eval(eval_gen, eval_metrics, zmodel):
    for tag, generator in eval_gen.items():

        genfun = generator.get_batch_generator()

        print('[%s]\t[Eval:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        res = dict([[k,0.] for k in eval_metrics.keys()])
        num_valid = 0
        for input_data, y_true in genfun:
            names = ['query', 'doc']
            shapes = [(None, 10), (None, 40)]
            list_input_data = _standardize_input_data(input_data, names, shapes, check_batch_axis=False)

            preprocessed_input_data = np.concatenate((list_input_data[0], list_input_data[1]), axis=1)

            y_pred = zmodel.forward(preprocessed_input_data)
            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                num_valid += 1
        generator.reset()
        i_e = 0
        print('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])), end='\n')
        sys.stdout.flush()

# Return  List(batch_input, batch_input, .....)
# there are totally 8995 pair in the dataset, and each time we would take a batch(100) samples
# roughly, set batch_num=100 would take the entire pairs.
def generate_training_data(train_gen, batch_num):
    zoo_input_data = []
    zoo_label = []
    count = 0
    while True:
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            for input_data, y_true_value in genfun:
                count += 1
                if count > batch_num:
                    return (zoo_input_data, zoo_label)
                names = ['query', 'doc']
                shapes = [(None, 10), (None, 40)]
                list_input_data = _standardize_input_data(input_data, names, shapes,
                                                          check_batch_axis=False)
                zoo_input_data.append(list_input_data)
                y_true_value = np.expand_dims(y_true_value, 1)
                zoo_label.append(y_true_value)

def train(config):

    print(json.dumps(config, indent=2), end='\n')
    # read basic config
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.%d'
    display_interval = int(global_conf['display_interval'])
    num_iters = int(global_conf['num_iters'])
    save_weights_iters = int(global_conf['save_weights_iters'])

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']


    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
            continue
        if 'text1_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
        if 'text2_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator( config = conf )

    ######### Load Model #########
    zmodel, kmodel = load_model(config)

    input = Input(name='input', shape=(2, 50))
    timeDistributed = TimeDistributed(layer = zmodel, input_shape=(2, 50))(input)
    z_knrm_model = Model(input=input, output=timeDistributed)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)

    epoch_num = 400
    batch_size = 200  # take a look at the config
    batch_num_per_epoch = 10
    #train_as_whole(z_knrm_model, zmodel, train_gen, eval_gen, eval_metrics)
    z_knrm_model.set_tensorboard("/tmp/matchzoo", "knrm-sgd-1e4")
    train_per_epoch(z_knrm_model, zmodel, train_gen, eval_gen, eval_metrics, optimMethod=SGD(1e-4))

    # train_per_epoch(z_knrm_model, zmodel, train_gen, eval_gen, eval_metrics, optimMethod=SGD(1e-4, leaningrate_schedule=Poly(0.5, 50 * 400)))
    #train_per_epoch(z_knrm_model, zmodel, train_gen, eval_gen, eval_metrics, optimMethod="adam")


def train_per_epoch(z_knrm_model, zmodel, train_gen, eval_gen, eval_metrics, epoch_num=400, batch_size=200, optimMethod=SGD(1e-4)):
    z_knrm_model.compile(optimizer=optimMethod, loss=zloss(batch=10))
    print('[Model] Model Compile Done.', end='\n')
    total_batches = 100
    total_samples = total_batches * batch_size
    # take 100 batch, each batch has 100 samples.
    # and each time the training process take 200 samples to assemble a batch for training
    # so there are 50 iteration for each epoch.
    (zoo_input_data, zoo_label) = generate_training_data(train_gen, batch_num=total_batches)
    new_zinput = preprocess(zoo_input_data)
    zoo_label = np.ones([int(total_samples/2), 2, 1])
    for i in range(0, epoch_num):
        z_knrm_model.fit(new_zinput, zoo_label, batch_size=200, nb_epoch=1, distributed=False)
        # z_knrm_model.saveModel('new_model_Adam.model', over_write=True)
        # zmodel.saveModel('zmodel.model', over_write=True)
        eval(eval_gen, eval_metrics, zmodel)

def train_as_whole(z_knrm_model, zmodel, train_gen, eval_gen, eval_metrics):
    z_knrm_model.compile(optimizer='adam', loss=zloss(batch=10))
    print('[Model] Model Compile Done.', end='\n')
    epoch_num = 400
    batch_num_per_epoch = 10
    batch_size = 200 # take a look at the config
    total_batches = epoch_num * batch_num_per_epoch
    total_samples = total_batches * batch_size
    (zoo_input_data, zoo_label) = generate_training_data(train_gen, batch_num=total_batches)
    new_zinput = preprocess(zoo_input_data)
    zoo_label = np.ones([int(total_samples/2), 2, 1])
    z_knrm_model.fit(new_zinput, zoo_label, batch_size=200, nb_epoch=1, distributed=False)
    z_knrm_model.saveModel('new_model_Adam.model', over_write=True)
    zmodel.saveModel('zmodel.model', over_write=True)
    eval(eval_gen, eval_metrics, zmodel)


def set_weights_per_layer(kmodel, zmodel, layer_name):
    klayer = kmodel.get_layer(layer_name)
    klayer_weights = klayer.get_weights()
    zlayer_weights = WeightsConverter.to_bigdl_weights(klayer, klayer_weights)
    zlayer = [l for l in zmodel.layers if l.name() == layer_name][0] # assert the result length is 1
    zlayer.set_weights(zlayer_weights)

def load_model(config):
    zmodel = load_zoo_model(config)
    # model.load_weights(weights_file)
    kmodel = load_keras2_model(config)

    ######## Get and Set Weights ########
    set_weights_per_layer(kmodel, zmodel, "embedding")
    set_weights_per_layer(kmodel, zmodel, "dense")

    return zmodel, kmodel

def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2), end='\n')
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])

    zmodel, kmodel = load_model(config)

    # test y_pred from zoo model and keras model
    # keras2_y_pred = kmodel.predict(input_data, batch_size=batch_size)
    # y_pred = model.forward(input_data)
    # # y_pred = model.predict(input_data, distributed=False)
    # equal = np.allclose(y_pred, keras2_y_pred, rtol=1e-5, atol=1e-5)
    # print(equal)
    # return y_pred

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    res = dict([[k,0.] for k in eval_metrics.keys()])

    # batch_size = 20
    # query_data = np.random.randint(0, 10000, [batch_size, 10])
    # doc_data = np.random.randint(0, 10000, [batch_size, 40])
    # input_data = [query_data, doc_data]
    # keras2_y_pred = keras2_model.predict(input_data, batch_size=batch_size)
    # y_pred = model.predict(input_data, distributed=False)
    # equal = np.allclose(y_pred, keras2_y_pred, rtol=1e-5, atol=1e-5)
    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print('[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        num_valid = 0
        res_scores = {}
        for input_data, y_true in genfun:
            ky_pred = kmodel.predict(input_data, batch_size=len(y_true))
            names = ['query', 'doc']
            shapes = [(None, 10), (None, 40)]
            list_input_data = _standardize_input_data(input_data, names, shapes, check_batch_axis=False)
           # list_input_data = [data[0:2, :] for data in list_input_data]
            # y_pred = zmodel.predict(list_input_data, distributed=False)
            y_pred = zmodel.forward(list_input_data)
            equal = np.allclose(y_pred, ky_pred, rtol=1e-5, atol=1e-5)
            print(equal)

            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts)-1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx+1]
                    for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y[1], t[1])
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s\tQ0\t%s\t%d\t%f\t%s\t%s\n'%(qid, did, inum, score, config['net_name'], gt))
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s %s %s %s\n'%(gt, qid, did, score))

        print('[Predict] results: ', '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()]), end='\n')
        sys.stdout.flush()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/knrm_wikiqa.config', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    if args.phase == 'train':
        train(config)
    elif args.phase == 'predict':
        predict(config)
    else:
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
