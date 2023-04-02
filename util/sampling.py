import numpy as np
import copy
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MinibatchSampling:
    def __init__(self, array, batch_size, sim):
        if len(array) < batch_size:
            raise Exception('Length of array is smaller than batch size. len(array): ' + str(len(array))
                            + ', batch size: ' + str(batch_size))

        self.array = copy.deepcopy(array)   # So that the original array won't be changed
        self.buffer = []
        self.buff_size = 0
        self.batch_size = batch_size
        self.flow_idx = 0 # 记录当前数据流的样本位置

        self.start_index = 0

        self.rnd_seed = (sim + 1) * 1000
        # np.random.RandomState(seed=self.rnd_seed).shuffle(self.array)

    def get_next_batch(self, current_bs):
        if self.start_index + current_bs >= len(self.buffer):
            self.rnd_seed += 1
            np.random.RandomState(seed=self.rnd_seed).shuffle(self.buffer)
            self.start_index = 0

        ret = [self.buffer[i] for i in range(self.start_index, self.start_index + current_bs)]
        self.start_index += current_bs

        return ret

    def init_buffer(self, init_size, buff_size):
        self.buffer = copy.deepcopy(self.array[0:init_size])
        self.buff_size = buff_size
        self.flow_idx = init_size
        np.random.RandomState(seed=1).shuffle(self.buffer)

    def buffer_update(self, update_val): # Unlimited buffer
        orig_buffer_len = len(self.buffer)
        new_buffer_len = orig_buffer_len + update_val
        self.buffer = copy.deepcopy(self.array[0:new_buffer_len])
        self.flow_idx += update_val

    def buffer_random_update(self, update_val): # Random sampling
        count = 0
        if len(self.buffer) < self.buff_size:
            tmp_val = update_val + len(self.buffer)
            if tmp_val <= self.buff_size:
                self.buffer = copy.deepcopy(self.array[0:tmp_val])
                self.flow_idx = tmp_val
                update_val = 0
            else:
                update_val = update_val - (self.buff_size - len(self.buffer))
                self.buffer = copy.deepcopy(self.array[0:self.buff_size])
                self.flow_idx = self.buff_size
        for idx in range(self.flow_idx, self.flow_idx + update_val):
            m = int(np.random.randint(0, self.buff_size, 1))
            self.buffer[m] = self.array[idx]
            count += 1
        self.flow_idx += update_val
        print("Random Sampling updates ", count, "samples in the buffer.")

    def buffer_res_update(self,update_val): # Limited buffer - Reservoir Sampling
        count = 0
        if len(self.buffer) < self.buff_size:
            tmp_val = update_val + len(self.buffer)
            if tmp_val <= self.buff_size:
                self.buffer = copy.deepcopy(self.array[0:tmp_val])
                self.flow_idx = tmp_val
                update_val = 0
            else:
                update_val = update_val - (self.buff_size - len(self.buffer))
                self.buffer = copy.deepcopy(self.array[0:self.buff_size])
                self.flow_idx = self.buff_size
        for idx in range(self.flow_idx, self.flow_idx + update_val):
            m = int(np.random.randint(0,idx+1,1))
            if m < self.buff_size:
                self.buffer[m] = self.array[idx]
                count += 1
        self.flow_idx += update_val
        print("Reservoir Sampling updates ", count, "samples in the buffer.")

    def buffer_fifo_update(self,update_val):
        tmp_val = self.flow_idx + update_val
        start_val = tmp_val - self.buff_size
        if start_val >= 0:
            self.buffer = copy.deepcopy(self.array[start_val:tmp_val])
        else:
            self.buffer = copy.deepcopy(self.array[0:tmp_val])
        self.flow_idx = tmp_val
        print("FIFO Sampling updates ", update_val, "samples in the buffer.")

    def get_flow_idx(self):
        return self.flow_idx

    def get_buffer(self):
        return self.buffer