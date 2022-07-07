import torch.nn as nn
import torch


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=-100, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.criterion_xent = None
        self.criterion_mse = None
        self.beta = 0.9999  # for class balanced loss

    def build_loss(self, mode='ce'):
        loss_func = None
        """Choices: ['ce' | 'focal' | 'ndvi' | 'batch']"""
        if mode == 'ce':
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            loss_func = self.cross_entropy_loss
        elif mode == 'focal':
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            loss_func = self.focal_loss
        elif mode == 'ndvi':
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            self.criterion_mse = nn.MSELoss()  # Default reduction: 'mean' (reduction='sum')
            # self.criterion_mse = nn.L1Loss()
            loss_func = self.ndvi_loss
        elif mode == 'batch':
            # weights are computed inside a batch:
            # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
            self.criterion_xent = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            loss_func = self.class_balanced_loss
        else:
            raise NotImplementedError

        if self.cuda:
            self.criterion_xent = self.criterion_xent.cuda()
            if self.criterion_mse is not None:
                self.criterion_mse = self.criterion_mse.cuda()

        return loss_func

    def cross_entropy_loss(self, logit, target):
        print(logit.shape)
        print(target.shape)
        n = logit.size()[0]

        loss = self.criterion_xent(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def focal_loss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()

        logpt = -self.criterion_xent(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def ndvi_loss(self, ndvi_features, logit, target_cls, ndvi_target, samples_per_cls, weight_mse=1.0):
        n, c, h, w = logit.size()
        assert (not torch.isnan(ndvi_target).any())

        # Effective Number of Samples (ENS)
        if not samples_per_cls == None:
            assert (samples_per_cls.shape[0] == c)
            weights = (1.0 - self.beta) / (1.0 - torch.pow(self.beta, samples_per_cls.float()))
            weights[weights == float('inf')] = 0
            weights = weights / torch.sum(weights) * c  # wights in the range [0, c]
            self.criterion_xent.weight = weights

        loss_xent = self.criterion_xent(logit, target_cls.long())
        loss_mse = self.criterion_mse(ndvi_features, ndvi_target)
        loss_mse *= weight_mse
        loss = loss_xent + loss_mse

        if self.batch_average:
            loss /= n

        return loss

    def class_balanced_loss(self, logit, target, samples_per_cls, weight_type='ENS'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks."""
        # Starting point:
        # https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
        n, c, h, w = logit.size()
        # beta = (self.total_num_samples - 1) / self.total_num_samples

        assert(samples_per_cls.shape[0] == c)

        # compute weights for each minibatch
        if weight_type == 'ENS':
            # Effective Number of Samples (ENS)
            weights = (1.0 - self.beta) / (1.0 - torch.pow(self.beta, samples_per_cls.float()))
            weights[weights == float('inf')] = 0
        elif weight_type == 'ISNS':
            # Inverse of Square Root of Number of Samples (ISNS)
            weights = 1.0 / torch.sqrt(torch.tensor([2, 1000, 1, 20000, 500]).float())
        else:
            # Inverse of Number of Samples (INS)
            weights = 1.0 / torch.tensor([2, 1000, 1, 20000, 500]).float()

        weights = weights / torch.sum(weights) * c  # wights in the range [0, c]

        self.criterion_xent.weight = weights
        loss = self.criterion_xent(logit, target.long())
        # print("loss weights:", self.criterion_xent.weight)

        if self.batch_average:
            loss /= n

        return loss
