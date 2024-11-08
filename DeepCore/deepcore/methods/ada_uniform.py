import numpy as np
from .ada_coresetmethod import AdaCoresetMethod

class AdaUniform(AdaCoresetMethod):
    def __init__(self, dst_train, network, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, network, args, fraction, random_seed)
        # self.balance = balance
        self.balance = False
        self.replace = replace
        self.model = network
        self.n_train = len(dst_train)
        self.train_indx = np.arange(self.n_train)

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""
        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        # c stands for class here
        for c in range(self.num_classes):
            
            # c_index = (self.dst_train.targets == c)
            
            # when working with tl
            c_index = self.train_indx[self.dst_train.tensors[1] == int(c)]
            # c_index = (torch.stack(self.dst_train.targets) == c)
            
            # print(len(c_index))
            # print(round(self.fraction * c_index.sum().item()))
            
            self.index = np.append(self.index,
                                   np.random.choice(all_index[c_index], round(self.fraction * len(c_index)),
                                                    replace=self.replace))
        return self.index

    def select_no_balance(self):
        np.random.seed(self.random_seed)
        # print(self.n_train)
        # print(round(self.n_train * self.fraction))
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return  self.index

    def select(self, **kwargs):
        return {"indices": self.select_balance() if self.balance else self.select_no_balance()}
