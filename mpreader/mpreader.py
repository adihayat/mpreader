#!/usr/bin/env python
import numpy as np
import os
import re
import Queue as queue
from collections import defaultdict
from data_queue import  DataQueue
import math
import multiprocessing as mp
import time
import warnings

class DataSource(object):

        def __init__(self, sampler , batch_size, data_size_dict  , reader,  num_workers=4):
            """
            @params:    sampler()       : return list of batch size lists of sample meta data (to be used by the reader)
                                          i.e [ [batch_size list of samples],
                                                [batch_size list of samples] ... ]
                        batch_size      : int batch size
                        data_size_dict  : buffer name , shape and type
                                            #                 buffer name           ,   buffer_shape, buffer_type
                                            data_size_dict = {"data"                : ( [224,640]   , np.uint8)     ,
                                                              "labels"              : ( [1]         , np.uint16)    }
                        reader(meta)    : return meta , buffers(meta) ;
                                          buffers is dictionary of <buffer_name : np.array> , np.array size and type is according to data_size_dict
                        num_workers     : num of working processes

            """
            self.num_workers    = num_workers
            self.sampler        = sampler
            self.batch_size     = batch_size
            self.move_close     = False
            self.batch_queue    = None
            self.reader         = reader
            self.sampler        = sampler
            self.workers        = []
            self.iter           = None

            self.data_shapes_templates = {}
            for k in data_size_dict:
                self.data_shapes_templates[k] = np.zeros( [batch_size] +  data_size_dict[k][0] , data_size_dict[k][1])


        def close(self):
            self.move_close = True
            try:
                self.iter.next()
            except Exception as E:
                pass


        def iterator(self):
            if not self.iter is None:
                warnings.warn("Iterator already grabbed")

            self.iter = self._iterator()
            return self.iter

        def _iterator(self):
            """
            returns: iterator to <buffers dict , metadata , batch_idx (in epoch) >
            """
            #-----------------------------------------------------------------------
            def process_samples(samples):
                data_samples   = defaultdict(list)
                meta = []
                for s in samples:
                    metadata , data = self.reader(s)
                    for k,v in data.iteritems():
                        data_samples[k].append(v)
                    meta.append(metadata)

                for k in data_samples:
                    data_samples[k] = np.array(data_samples[k] , dtype=self.data_shapes_templates[k].dtype)

                return data_samples, meta

            #-----------------------------------------------------------------------
            def batch_producer(sample_queue, batch_queue):
                while True:
                    #---------------------------------------------------------------
                    # Process the sample
                    #---------------------------------------------------------------
                    try:
                        samples = sample_queue.get(timeout=0.1)
                    except Exception as E:
                        time.sleep(0.01)
                        continue

                    data , metadata = process_samples(samples)

                    #---------------------------------------------------------------
                    # Pad the result in the case where we don't have enough samples
                    # to fill the entire batch
                    #---------------------------------------------------------------
                    if data.values()[0].shape[0] < self.batch_size:
                        assert(0)
                    else:
                        batch_queue.put(data, metadata , timeout=0.1)
            #-----------------------------------------------------------------------
            #-------------------------------------------------------------------
            # Set up the parallel generator
            #-------------------------------------------------------------------
            #---------------------------------------------------------------
            # Set up the queues
            #---------------------------------------------------------------

            max_size = self.num_workers*10
            sample_queue = mp.Queue()
            self.batch_queue = DataQueue(self.data_shapes_templates, max_size)

            #---------------------------------------------------------------
            # Set up the workers. Make sure we can fork safely even if
            # OpenCV has been compiled with CUDA and multi-threading
            # support.
            #---------------------------------------------------------------
            for i in range(self.num_workers):
                args = (sample_queue, self.batch_queue)
                w = mp.Process(target=batch_producer, args=args)
                self.workers.append(w)
                w.daemon = True
                w.start()


            while not self.move_close:

            #---------------------------------------------------------------
            # Fill the sample queue with data
            #---------------------------------------------------------------

                batch_samples = self.sampler()

                for samples in batch_samples:
                    sample_queue.put(samples)

                #---------------------------------------------------------------
                # Return the data
                #---------------------------------------------------------------
                for batch_idx in xrange(len(batch_samples)):
                    if self.move_close :
                        break

                    data , metadata = self.batch_queue.get()
                    num_items = len(metadata)
                    yield data, metadata , batch_idx

            #---------------------------------------------------------------
            # Join the workers
            #---------------------------------------------------------------
            try :
                self.batch_queue.close()
            except Exception as E:
                pass

            try :
                while not sample_queue.empty():
                    sample_queue.get()
            except Exception as E:
                pass

            try :
                sample_queue.close()
            except Exception as E:
                pass

            for w in self.workers:
                w.terminate()


