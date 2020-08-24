from fastai2.basics import *

class SkipMetricPartException(Exception):
  pass

class AvgPartMetric(AvgMetric):
  "Average the values of `func` allowing to raise exception and omit accumulation"
  def accumulate(self, learn):
    try:
      val, bs = self.func(learn.pred, *learn.yb)
      self.total += to_detach(val)*bs
      self.count += bs
    except SkipMetricPartException: pass
    except Exception as e: print(e)

def accuracy_with_na(inp, targ, thresh=0.4, na_idx=0, axis=-1, sigmoid=True):
  "Compute accuracy assuming that prediction below threshold belongs to na"
  if sigmoid: inp = inp.sigmoid()
  valm, argm = inp.max(dim=axis)
  argm[valm < thresh] = 0 # treat all values below threshold as first category (#Na#)
  inp, targ = flatten_check(argm, targ)
  return (inp==targ).float().mean()

def _accuracy_without_na(inp, targ, na_idx=0, axis=-1):
  "Compute accuracy with `targ` when `pred` omiting #na# category"
  idxs = targ!=na_idx
  if idxs.any():
    inp, targ = flatten_check(inp[idxs].argmax(dim=axis), targ[idxs])
    return (inp==targ).float().mean(), targ.shape[axis]
  else:
     # skip accumulating metric if there is only #na# category in batch
    raise SkipMetricPartException

_accuracy_without_na.__name__ = 'accuracy_without_na'
accuracy_without_na = AvgPartMetric(_accuracy_without_na)
