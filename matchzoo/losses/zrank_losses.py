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