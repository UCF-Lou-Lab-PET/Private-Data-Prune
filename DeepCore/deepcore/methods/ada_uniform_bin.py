import numpy as np
from .ada_coresetmethod import AdaCoresetMethod
import time

# select in batch (in block)

class AdaUniformBin(AdaCoresetMethod):
    def __init__(self, dst_train, network, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, network, args, fraction, random_seed)
        # self.balance = balance
        self.balance = False
        self.replace = replace
        self.model = network
        self.n_train = len(dst_train)
        self.train_indx = np.arange(self.n_train)

    def select_no_balance(self):
        # np.random.seed(self.random_seed)
        np.random.seed(int(time.time()))
        
        block_size = self.args.train_batch
        num_blocks = int(np.floor((self.n_train * self.fraction) / block_size))  # Use ceiling to round up
        print(num_blocks)
        max_start_index = self.n_train - block_size 
        possible_starts = np.arange(0, max_start_index + 1, block_size)
        # np.random.shuffle(possible_starts)
        
        if len(possible_starts) < num_blocks:
            raise ValueError("Not enough possible starts to form the required number of blocks.")

        starting_indices = possible_starts[:num_blocks] 
        # hand_index = [80, 50, 60]
        # starting_indices = [possible_starts[i] for i in hand_index[:num_blocks]] 

        # Generate the blocks of indices
        bin_index = np.concatenate([np.arange(start, start + block_size) for start in starting_indices])
        
        # hand_start = 50
        # bin_index = np.arange(hand_start, hand_start+int((self.n_train * self.fraction)))
        # print(self.n_train)
        # print(round(self.n_train * self.fraction))
        self.index = bin_index
        # print(bin_index)
        print(len(bin_index))

        return  self.index

    def select(self, **kwargs):
        return {"indices": self.select_no_balance()}
