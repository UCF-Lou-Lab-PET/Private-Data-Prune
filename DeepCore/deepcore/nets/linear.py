import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder


class Linear(nn.Module):
    def __init__(self, in_dim, num_classes, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(Linear, self).__init__()
        self.fc_1 = nn.Linear(in_dim, num_classes)

                # Custom initialization
        nn.init.xavier_uniform_(self.fc_1.weight)
        # nn.init.zeros_(self.fc_1.weight)
        if self.fc_1.bias is not None:
            nn.init.constant_(self.fc_1.bias, 0)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_1

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            # out = x.view(x.size(0), -1)
            out = x
            out = self.embedding_recorder(out)
            out = self.fc_1(out)
        return out