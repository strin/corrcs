import sys import os
import numpy as np
sys.path.append('../../')
from utils.sampler import *
import scipy.io as sio

def observe_polling(src, dest, ratio):
  """
    observe_polling: generate observation data given measurement ratio.
      > input
        * src: polling.mat
        * dest: output path
        * ratio: measurement ratio, between (0,1].
  """
  mat = sio.loadmat(src)
  bgraph = toInt(mat['graph'] > 0.9 * abs(mat['graph']).max())
  data = mat['data']
  mask = sampleByRow(data, ratio)
  mat.update({'mask':mask, 'bgraph':bgraph, 'm_ratio': ratio})
  path = dest
  os.system('mkdir -p ' + path)
  path += '/ratio_%s.mat' % str(ratio)
  sio.savemat(path, mat, do_compression=True)


