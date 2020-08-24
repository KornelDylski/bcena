from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *
from torch import nn

class BCENaLoss(nn.Module):
  "BCE loss function with first category treated as unknown (#na#) and zeroing it"
  y_int = True

  def __init__(self, logits=True, reduction='mean'):
    super().__init__()
    self.reduction = reduction
    self.logits = logits

  def forward(self, input, target):
    target = F.one_hot(target, input.shape[1]).float()
    target[:, 0] = 0 # first category is #na# category so it should be zeroed
    if self.logits:
      return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction) # sigmoid + bce
    else:
      return F.binary_cross_entropy(input, target, reduction=self.reduction) # bce


@delegates(keep=True)
class BCENaLossFlat(BaseLoss):
  "Same as `BCENaLoss`, but flattens input and target and decodes as first (#na#) category if below threshold"
  def __init__(self, *args, axis=-1, thresh=0.5, **kwargs):
    super().__init__(BCENaLoss, *args, axis=axis, **kwargs)
    self.thresh = thresh

  def decodes(self, x):
    valm, argm = x.max(dim=self.axis)
    argm[valm < self.thresh] = 0
    return argm

  def activation(self, x): return F.sigmoid(x)


@delegates()
class BCEWithLogitsLossOneHotFlat(BCEWithLogitsLossFlat):
  "BCEWithLogitsLoss with one-hot encoding before, useful for single-label classification with BCE"
  def __call__(self, inp, targ, **kwargs):
    return super().__call__(inp, F.one_hot(targ, inp.shape[1]), **kwargs)
  def decodes(self, x):    return x.argmax(dim=-1)
  def activation(self, x): return F.sigmoid(x)