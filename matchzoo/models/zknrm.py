# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import zoo.pipeline.api.autograd as A
import zoo.pipeline.api.keras.layers as layer1
from zoo.pipeline.api.keras2.layers import *
from zoo.pipeline.api.keras.models import *
from matchzoo.models.model import BasicModel
from matchzoo.utils.utility import *
from bigdl.keras.converter import WeightsConverter

class ZKNRM(BasicModel):
    def __init__(self, config):
        super(ZKNRM, self).__init__(config)
        self._name = 'ZKNRM'
        self.check_list = [ 'text1_maxlen', 'kernel_num','sigma','exact_sigma',
                            'embed', 'embed_size', 'vocab_size']
        self.setup(config)
        if not self.check():
            raise TypeError('[ZKNRM] parameter check wrong')
        print('[ZKNRM] init done')

    def setup(self, config):
        self.set_default('kernel_num', 11)
        self.set_default('sigma', 0.1)
        self.set_default('exact_sigma', 0.001)
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.config.update(config)

    def share(self, layer, in1, in2):
        merged = layer1.merge([in1, in2], mode="concat", concat_axis=1)
        shared_out = layer(merged)
        out1 = shared_out.slice(1, 0, 10)
        out2 = shared_out.slice(1, 10, 40)
        return out1, out2

    def build(self):
        def Kernel_layer(mu,sigma):
            def kernel(x):
                return A.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return A.Lambda(lambda x: kernel(x)) #Activation(kernel)

        input = klayers1.Input(name='input', shape=(50,))
        query = input.slice(1, 0, 10) #klayers1.Input(name='query', shape=(self.config['text1_maxlen'],))
        #show_layer_info('Input', query)
        doc = input.slice(1, 10, 40) #klayers1.Input(name='doc', shape=(self.config['text2_maxlen'],))
        #show_layer_info('Input', doc)
        embedding = klayers1.Embedding(self.config['vocab_size'], self.config['embed_size'],
                              name="embedding") # weights=[self.config['embed']], trainable=self.config['train_embed'],
        q_embed, d_embed = self.share(embedding, query, doc)
        #show_layer_info('Embedding', q_embed)
        #show_layer_info('Embedding', d_embed)
        mm = A.batch_dot(q_embed, d_embed, axes=[2, 2], normalize=False)
        #show_layer_info('Dot', mm)

        KM = []
        for i in range(self.config['kernel_num']):
            mu = 1. / (self.config['kernel_num'] - 1) + (2. * i) / (self.config['kernel_num'] - 1) - 1.0
            sigma = self.config['sigma']
            if mu > 1.0:
                sigma = self.config['exact_sigma']
                mu = 1.0
            mm_exp = Kernel_layer(mu, sigma)(mm)
            #show_layer_info('Exponent of mm:', mm_exp)
            mm_doc_sum = A.Lambda(lambda x: A.sum(x, 2))(mm_exp) #reduce_sum(x, 2)
            #show_layer_info('Sum of document', mm_doc_sum)
            mm_log = A.Lambda(lambda x: A.log(x + 1.0))(mm_doc_sum)
            #show_layer_info('Logarithm of sum', mm_log)
            mm_sum = A.Lambda(lambda x: A.sum(x, 1))(mm_log)
            #show_layer_info('Sum of all exponent', mm_sum)
            KM.append(mm_sum)

        KMStack = A.stack(KM, 1)
        Phi = A.Lambda(lambda x: x)(KMStack)
        #show_layer_info('Stack', Phi)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax', bias_initializer='zeros', name="dense")(Phi)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1, bias_initializer='zero', name="dense")(Phi)
        #show_layer_info('Dense', out_)

        #timedistributed = layer1.TimeDistributed(out_, input_shape=(10, 12))
        model = Model(input=input, output=[out_])
        return model

