import mpreader
import numpy as np

#                 buffer name           ,   buffer_shape, buffer_type
data_size_dict = {"data"                : ( [224,640]   , np.uint8)     ,
                  "labels"              : ( [1]         , np.uint16)    }

# Example simple reader, reading single sample at a time
class Reader(object):
    # Reader init , opening files etc...
    def __init__(self , path , data_size_dict):
        self.files      = {}
        self.data_keys  = data_size_dict.keys()
        for key in data_size_dict:
            self.files[key] = np.memmap(path + "." + key + ".bin" , data_size_dict[key][1] , mode='r').reshape([-1] + data_size_dict[key][0])

    # Reading single sample indicated by <meta>
    def __call__(self , meta):
        buffers = {}
        for k in self.data_keys:
            buffers[k] = self.files[k][meta]

        return meta , buffers

# Example simple sampler, sampling each epoch
class Sampler(object):
    # Register initial samples list
    def __init__(self, samples_list , batch_size):
        self.sample_list  = samples_list
        self.batch_size   = batch_size


    # Return list of batch_size samples
    def __call__(self):
        samples = []
        for offset in range(0, len(self.sample_list), self.batch_size):
            if (offset + self.batch_size) > len(self.sample_list):
                samples.append(self.sample_list[-self.batch_size : ])
            else:
                samples.append(self.sample_list[offset:offset+self.batch_size])
        return samples


if __name__ == "__main__":

    import tensorflow as tf
    import dataset_utils

    samples_list = np.arange(1000).tolist()
    ds = mpreader.DataSource(Sampler(samples_list , 16) , 16  ,data_size_dict ,
                             Reader("/path/to/binary" , data_size_dict ))

    iterator = ds.iterator()
    dataset = tf.data.Dataset.from_generator(lambda : dataset_utils.wrap_iter_data(iterator) , dataset_utils.getTFTypeDict(data_size_dict) ,
                                             dataset_utils.getTFShapeDict(data_size_dict))
    diter = dataset.make_initializable_iterator()
    el = diter.get_next(name="input")
    import time
    print el['data'].get_shape()

    num_of_trails = 1000

    duration = 0

    with tf.Session() as sess:
        sess.run(diter.initializer)

        for k in xrange(num_of_trails):
            start = time.time()
            [_el] = sess.run([el])
            end   = time.time()
            duration += end - start
            time.sleep(0.01)

    print "fetch_time = {}".format((duration)/num_of_trails)
    print _el['data'].shape
    print _el['labels'].shape
    print _el['idx']

    ds.close()


