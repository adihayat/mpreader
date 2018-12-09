import tensorflow as tf
import numpy as np


def getTFTypeDict(data_size_dict):
    """
    producing compatible tf.types dict from data_size_dict
    @params:

                data_size_dict  : buffer name , shape and type
                                    #                 buffer name           ,   buffer_shape, buffer_type
                                    data_size_dict = {"data"                : ( [224,640]   , np.uint8)     ,
                                                      "labels"              : ( [1]         , np.uint16)    }

    """
    def dtypeConvert(np_dtype):
        if (np_dtype == np.uint8) or( np_dtype == np.uint16) :
            return tf.int32
        if np_dtype == np.float32 :
            return tf.float32

    rdict = {}
    for k , v in data_size_dict.iteritems():
        rdict[k] = dtypeConvert(v[1])


    assert(not 'meta' in data_size_dict.keys())
    assert(not 'idx' in data_size_dict.keys())

    rdict['meta'] = tf.string
    rdict['idx']  = tf.int32

    return rdict

def getTFShapeDict(data_size_dict):
    """
    producing compatible tf.types dict from data_size_dict
    @params:

                data_size_dict  : buffer name , shape and type
                                    #                 buffer name           ,   buffer_shape, buffer_type
                                    data_size_dict = {"data"                : ( [224,640]   , np.uint8)     ,
                                                      "labels"              : ( [1]         , np.uint16)    }

    """
    rdict = {}
    for k , v in data_size_dict.iteritems():
        # Adding None for unknown batch size
        rdict[k] = [None] +  v[0]


    assert(not 'meta' in data_size_dict.keys())
    assert(not 'idx' in data_size_dict.keys())

    rdict['meta'] = ()
    rdict['idx']  = ()

    return rdict


def wrap_iter_data(iterator):
    """
    Adding Meta , idx entries to returned dict
    @params:
        iterator: base iterator to be wrapped
    """
    while True:
        e  = iterator.next()
        e[0]['meta'] = str(e[1])
        e[0]['idx']  = e[2]
        yield e[0]
