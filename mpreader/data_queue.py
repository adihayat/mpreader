#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   17.09.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import queue as q
import numpy as np
import multiprocessing as mp
import time

#-------------------------------------------------------------------------------
class DataQueue:
    #---------------------------------------------------------------------------
    def __init__(self, data_templates, maxsize):
        #-----------------------------------------------------------------------
        # Figure out the data tupes, sizes and shapes of both arrays
        #-----------------------------------------------------------------------
        self.data_queue = {}
        self.array_pool = []
        self.array_queue = mp.Queue(maxsize)
        self.move_close  = False
        for k in data_templates:
            self.data_queue[k] = dict(shape=data_templates[k].shape , dtype=data_templates[k].dtype , data_bc=len(data_templates[k].tobytes())  , array_pool=[])
            for i in xrange(maxsize):
                data_buff = mp.Array('c', self.data_queue[k]['data_bc'], lock=False)
                data_arr = np.frombuffer(data_buff, dtype=self.data_queue[k]['dtype'])
                data_arr = data_arr.reshape(self.data_queue[k]['shape'])
                self.data_queue[k]['array_pool'].append(data_arr)

        for i in xrange(maxsize):
            self.array_queue.put(i)


        self.queue = mp.Queue(maxsize)



    def __del__(self):
        self.close()


    def close(self):
        self.move_close = True

    #---------------------------------------------------------------------------
    def put(self, data, gt_params, *args, **kwargs):
        #-----------------------------------------------------------------------
        # Check whether the params are consistent with the data we can store
        #-----------------------------------------------------------------------
        def check_consistency(name, arr, dtype, shape, byte_count):
            if type(arr) is not np.ndarray:
                raise ValueError(name + ' needs to be a numpy array')
            if arr.dtype != dtype:
                raise ValueError('{}\'s elements need to be of type {} but is {}' \
                                 .format(name, str(dtype), str(arr.dtype)))
            if arr.shape != shape:
                raise ValueError('{}\'s shape needs to be {} but is {}' \
                                 .format(name, shape, arr.shape))
            if len(arr.tobytes()) != byte_count:
                raise ValueError('{}\'s byte count needs to be {} but is {}' \
                                 .format(name, byte_count, len(arr.data)))


        for k , v in data.iteritems():
            check_consistency(k , v , self.data_queue[k]['dtype'] , self.data_queue[k]['shape'] , self.data_queue[k]['data_bc'])

        #-----------------------------------------------------------------------
        # If we can not get the slot within timeout we are actually full, not
        # empty
        #-----------------------------------------------------------------------
        while not self.move_close:
            try:
                arr_id = self.array_queue.get(*args, **kwargs)
                break
            except q.Empty:
                time.sleep(0.1)
                continue

        if self.move_close:
            return

        #-----------------------------------------------------------------------
        # Copy the arrays into the shared pool
        #-----------------------------------------------------------------------
        for k , v in data.iteritems():
            self.data_queue[k]['array_pool'][arr_id][...] = v

        self.queue.put((arr_id, gt_params), *args, **kwargs)

    #---------------------------------------------------------------------------
    def get(self, *args, **kwargs):
        item = self.queue.get(*args, **kwargs)
        arr_id = item[0]
        gt_params = item[1]

        data = {}
        for k in self.data_queue:
            data[k] = np.copy(self.data_queue[k]['array_pool'][arr_id])

        self.array_queue.put(arr_id)

        return data, gt_params

    #---------------------------------------------------------------------------
    def empty(self):
        return self.queue.empty()
