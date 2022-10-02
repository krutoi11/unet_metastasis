import torch
import torch.nn.functional as F

"""
 def loss(self, y_pred, y_true):
     assert y_pred.size() == y_true.size()
     y_pred = y_pred[:, 0].contiguous().view(-1)
     y_true = y_true[:, 0].contiguous().view(-1)
     intersection = (y_pred * y_true).sum()
     dsc = (2. * intersection + self.smooth) / (
             y_pred.sum() + y_true.sum() + self.smooth
     )
     return 1. - dsc

 """


def weighted_bce(output, target):
    weights = [100, 1]
    loss = weights[0] * (target * torch.log(output + 1e-3)) + \
           weights[1] * ((1 - target) * torch.log(1 - output + 1e-3))

    return torch.neg(torch.mean(loss))


def iou(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def dice(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


def focal_loss(inputs, targets, alpha=0.8, gamma=2, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # first compute binary cross-entropy
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

    return focal_loss