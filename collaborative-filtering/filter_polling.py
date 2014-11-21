from filter import *

import sys, os
sys.path.append('../')
from utils import sampler

import scipy.io as sio
import numpy as np
import numpy.random as npr
import numpy.linalg as npla

def filter_window(path_in):
  """ 
    filter_window:
      perform collaborative filtering on a sliding window
  """
  mat = sio.loadmat(path_in)
  m = int(mat['m'])
  n = int(mat['n'])
  dim = int(mat['dim'])
  X = mat['X']
  mask = mat['mask']
  data = mat['data']

  nowX = list()
  nowMask = list()
  for j in range(m-dim+1):
    nowX.append(X[:, range(j,X.shape[1], m-dim+1)])
    nowMask.append(mask[j:j+dim, :])

  res = list()
  for j in range(m-dim+1):
    print j
    arr = np.transpose(nowX[j])        # treat features as users.
    mask = np.transpose(nowMask[j])    # treat rows as movies.
    dic = sampler.arrToLists(arr, mask)
    rows = sampler.toStr(range(arr.shape[0]))
    columns = sampler.toStr(range(arr.shape[1]))
    recon = recommend(dic, rows, columns)
    recon = sampler.listsToArr(recon, arr.shape)
    res.append(recon)

  reX_g = np.zeros(X.shape)
  for j in range(m-dim+1):
    reX_g[:, range(j, X.shape[1], m-dim+1)] = res[j]

  data_c_g = np.zeros(data.shape)
  data_re_g = np.zeros(data.shape)
  for ni in range(n):
    for j in range(m-dim+1):
      data_c_g[range(j, j+dim), ni] += 1
      data_re_g[range(j, j+dim), ni] += reX_g[:, ni * (m-dim+1) + j]

  data_re_g = np.divide(data_re_g, data_c_g)
  mse = npla.norm(data_re_g - data, 'fro') ** 2 / data.shape[0] / data.shape[1]
  print 'mse', mse

def filter(path_in, path_out, filename):
  mat = sio.loadmat(path_in + '/' + filename)
  m = int(mat['m'])
  n = int(mat['n'])
  dim = int(mat['dim'])
  X = mat['X']
  mask = mat['mask']
  arr = mat['data']
  ratio = mat['m_ratio']
  
  dic = sampler.arrToLists(arr, mask)
  rows = sampler.toStr(range(arr.shape[0]))
  columns = sampler.toStr(range(arr.shape[1]))
  recon = recommend(dic, rows, columns)
  recon = sampler.listsToArr(recon, arr.shape)

  mse = npla.norm(recon - arr, 'fro') ** 2 / arr.shape[0] / arr.shape[1]
  print 'mse', mse

  os.system('mkdir -p ' + path_out)
  sio.savemat(path_out + '/' + filename, {'mse':mse, 'recon':recon, 'data':arr})

if __name__ == "__main__":
  filter('../data/scaum/polling/', '../result/scaum/polling/', 'ratio_0.5.mat')
