import sys, os
import numpy as np
sys.path.append('../../')
import utils.sampler as sampler
import utils.score
import collaborative_filtering.filter as cfilter
import scipy.io as sio

def process_ebola_casualty(file_in, file_out):
  """
    process raw ebola casualty data into matrix format.
  """
  lines = file(file_in).readlines()
  condition = lambda line: line[0] == "Cumulative number of confirmed Ebola cases" or line[0] == "Cumulative number of probable Ebola cases" \
                                or line[0] == "Cumulative number of suspected Ebola cases"
  shorten = {'Cumulative number of confirmed Ebola cases': 'confirmed|', 
             'Cumulative number of probable Ebola cases':'probable|',
             'Cumulative number of suspected Ebola cases':'suspected|'}
  data = dict()
  rows = dict()
  cols = dict()
  for line in lines:
    line = line.split(',')
    if condition(line):
      if not rows.has_key(line[1]):
        rows[line[1]] = len(rows)
      line[2] = shorten[line[0]] + line[2]
      if not cols.has_key(line[2]):
        cols[line[2]] = len(cols)
  table = np.ones((len(rows), len(cols))) * float('NaN')
  for line in lines:
    line = line.split(',')
    if condition(line):
      line[2] = shorten[line[0]] + line[2]
      table[rows[line[1]]][cols[line[2]]] = float(line[3])
  sio.savemat(file_out, {'data':table, 'rows':rows, 'cols':cols})

def observe_ebola_casualty(src, dest, ratio):
  """
    generate observation data given measurement ratio.
      > input
        * src: polling.mat
        * dest: output path
        * ratio: measurement ratio, between (0,1].
  """
  mat = sio.loadmat(src)
  data = mat['data']
  mask = sampler.sampleByCol(data, ratio)
  mat.update({'mask':mask, 'm_ratio': ratio})
  path = dest
  os.system('mkdir -p ' + path)
  path += '/ratio_%s.mat' % str(ratio)
  sio.savemat(path, mat, do_compression=True)

def filter_ebola_casualty(path_in, path_out, filename):
  mat = sio.loadmat(path_in + '/' + filename)
  mask = mat['mask']
  arr = mat['data']
  ratio = mat['m_ratio']
  
  dic = sampler.arrToLists(arr, mask)
  print dic
  rows = sampler.toStr(range(arr.shape[0]))
  columns = sampler.toStr(range(arr.shape[1]))
  recon = cfilter.recommend(dic, rows, columns)
  recon = sampler.listsToArr(recon, arr.shape)

  print recon
  mse = utils.score.mse(recon, arr)
  print 'mse', mse

  os.system('mkdir -p ' + path_out)
  sio.savemat(path_out + '/' + filename, {'mse':mse, 'recon':recon, 'data':arr})
