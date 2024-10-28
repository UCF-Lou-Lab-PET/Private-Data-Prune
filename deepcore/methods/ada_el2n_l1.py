import torch, time
import numpy as np
from .ada_coresetmethod import AdaCoresetMethod
from ..nets.nets_utils import MyDataParallel
from torch import nn
import datetime

class AdaEL2NL1(AdaCoresetMethod):
    def __init__(self, dst_train, network, args, fraction=0.5, random_seed=None, repeat=1,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, network, args, fraction, random_seed)
        self.epochs = 0
        self.n_train = len(dst_train)
        print("self n train = len dist train = ", self.n_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat
        self.model = network
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()
        self.model_optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)

        # self.balance = balance
        self.balance = False

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def finish_run(self):
        self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()

        embedding_dim = self.model.get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers)
        sample_num = self.n_train
        # l2_loss = torch.nn.MSELoss(reduction='none')
        l1_loss = torch.nn.L1Loss(reduction='none')

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).to(self.args.device)
            with torch.no_grad():
                self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num),
                self.cur_repeat] = l1_loss(one_hot_labels, probabilities).sum(dim=1)


        self.model.train()

        self.model.embedding_recorder.record_embedding = False


    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        self.before_run()

        return self.finish_run()


    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.args.device)

        print("called EL2N-l1, with {} ensemble and {} epochs, {} all samples".format(self.repeat, self.epochs, len(self.dst_train)))

        for self.cur_repeat in range(self.repeat):
            # will asign value yo norm matrix
            self.run()
            self.random_seed = self.random_seed + 5

        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][: self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.tensors[1] == int(c)]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])
        return {"indices": top_examples, "scores": self.norm_mean}
