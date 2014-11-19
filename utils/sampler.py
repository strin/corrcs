'''
sampling schemes for collecting data.
'''
import numpy as np
import numpy.random as npr

toInt = np.vectorize(int)
toStr = np.vectorize(str)

def sample(arr, ratio):
  """ sample.
      > input
        arr: 2-D array to be sampled. 
        ratio: percentage of entries sampled per row. 
      > output: a mask array of the same shape.
  """
  arr = np.array(arr)
  mask = toInt(npr.random(arr.shape) < ratio)
  return mask


def arrToLists(arr, mask):
  """ given an array, output its list representation.
      > input
        arr: 2-D array.
      > output: a dict.
  """
  mask = np.array(mask)
  output = dict()
  for i in range(mask.shape[0]):
    dic = dict()
    for j in range(mask.shape[1]):
      if mask[i][j] == 1:
        dic[str(j)] = arr[i][j]
    if len(dic) == 0:
      continue
    output[str(i)] = dic
  return output

def listsToArr(dic, shape):
  """ given lists, convert it into 2-D array.
      > input
        dic: a dict.
      > output: 2-D np.array
  """
  output = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      output[i][j] = dic[str(i)][str(j)]
  return output



if __name__ == "__main__":
  arr = [[1,2],[3,4]]
  mask = sample(arr, 0.5)
  print 'mask = ', mask
  dic = arrToLists(arr, mask)
  print 'dic = ', dic



