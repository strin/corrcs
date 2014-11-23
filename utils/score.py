import numpy as np

def mse(arr1, arr2):
  """ compute mean-square error, ignoring NaN entries
  """
  (arr1, arr2) = (np.array(arr1), np.array(arr2))
  assert(arr1.shape == arr2.shape)
  def sq(x):
    if x != x: # isnan
      return 0
    else:
      return x**2
  def count(x):
    if x != x: # isnan
      return 0
    else:
      return 1
  sq_v = np.vectorize(sq)
  count_v = np.vectorize(count)

  return sq_v(arr1-arr2).sum() / float(count_v(arr1-arr2).sum())

