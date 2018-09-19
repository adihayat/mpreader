#!/usr/bin/env python
import numpy as np
import os
import re
import Queue as queue
from collections import defaultdict
from data_queue import  DataQueue
import math
import multiprocessing as mp

class DataSource(object):

        def __init__(self, sampler , batch_size, data_size_dict  , reader,  num_workers=4):
            self.num_workers    = num_workers
            self.sampler        = sampler
            self.batch_size     = batch_size
            self.move_close     = False
            self.batch_queue    = None
            self.reader         = reader
            self.sampler        = sampler
            self.workers        = []

            self.data_shapes_templates = {}
            for k in data_size_dict:
                self.data_shapes_templates[k] = np.zeros( [batch_size] +  data_size_dict[k][0] , data_size_dict[k][1])


        def close(self):
            self.move_close = True
            for w in self.workers:
                w.terminate()


        def iterator(self):

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
                while not self.move_close:
                    #---------------------------------------------------------------
                    # Process the sample
                    #---------------------------------------------------------------
                    try:
                        samples = sample_queue.get(timeout=1)
                    except Exception as E:
                        print "WARNING empty queue"
                        print E
                        break

                    data , metadata = process_samples(samples)

                    #---------------------------------------------------------------
                    # Pad the result in the case where we don't have enough samples
                    # to fill the entire batch
                    #---------------------------------------------------------------
                    if data.values()[0].shape[0] < self.batch_size:
                        assert(0)
                    else:
                        batch_queue.put(data, metadata , timeout=1)
            #-----------------------------------------------------------------------
            #-------------------------------------------------------------------
            # Set up the parallel generator
            #-------------------------------------------------------------------
            #---------------------------------------------------------------
            # Set up the queues
            #---------------------------------------------------------------

            max_size = self.num_workers*5
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
                w.daemon = False
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
                for offset in xrange(len(batch_samples)):
                    if self.move_close :
                        break

                    data , metadata = self.batch_queue.get()
                    num_items = len(metadata)
                    yield data, metadata

            #---------------------------------------------------------------
            # Join the workers
            #---------------------------------------------------------------
            for w in self.workers:
                w.join()


