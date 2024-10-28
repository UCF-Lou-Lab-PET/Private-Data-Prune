import numpy as np
from .ada_coresetmethod import AdaCoresetMethod


class AdaFull(AdaCoresetMethod):
    def __init__(self, dst_train, network, args, fraction, random_seed, **kwargs):
        self.n_train = len(dst_train)

    def select(self, **kwargs):
        return {"indices": np.arange(self.n_train)}
