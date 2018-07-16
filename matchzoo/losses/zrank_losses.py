from __future__ import print_function

import zoo.pipeline.api.autograd as A
from zoo.pipeline.api.keras.layers import *
from keras.utils.generic_utils import deserialize_keras_object

mz_specialized_losses = {'rank_hinge_loss', 'rank_crossentropy_loss'}

def serialize(rank_loss):
    return rank_loss.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')

def rank_hinge_loss(kwargs=None):
    margin = 1.
    if isinstance(kwargs, dict) and 'margin' in kwargs:
        margin = kwargs['margin']

    def _margin_loss(y_true, y_pred):
        # output_shape = K.int_shape(y_pred)
        t1 = []
        for y in range(0, batch, 2):
            x = y_pred.slice(0, y, 1)
            t1.append(x)
        y_pos = merge(t1, mode="concat", concat_axis=0)
        t2 = []
        for y in range(1, batch, 2):
            x = y_pred.slice(0, y, 1)
            t2.append(x)
        y_neg = merge(t2, mode="concat", concat_axis=0)
        loss = A.maximum(0., margin + y_neg - y_pos)
        return loss
    return _margin_loss

def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)