import sys, os
sys.path.append('../')
from utils import sampler

import scipy.io as sio
import numpy as np
import numpy.random as npr
import numpy.linalg as npla

from scikits.crab import datasets
from scikits.crab.recommenders.knn.item_strategies import AllPossibleItemsStrategy, ItemsNeighborhoodStrategy
from scikits.crab.recommenders.svd.classes import MatrixFactorBasedRecommender
from scikits.crab.models.classes import  MatrixPreferenceDataModel, MatrixBooleanPrefDataModel

def recommend(data, users, items):
  # print 'data = ', data
  matrix_model = MatrixPreferenceDataModel(data)

  items_strategy = AllPossibleItemsStrategy()
  # items_strategy = ItemsNeighborhoodStrategy()
  recsys = MatrixFactorBasedRecommender(
          model=matrix_model,
          items_selection_strategy=items_strategy,
          n_features=2)

  score = dict()
  for user in users:
    row = dict()
    for item in items:
      if data.has_key(user) and data[user].has_key(item):     # the user has already rated.
        row[item] = data[user][item]
      else:
        row[item] = recsys.estimate_preference(user, item)
    score[user] = row

  # print 'score = ', score
  return score

if __name__ == "__main__":
  data = sio.loadmat('../data/scaum/scaum-611-normalized.mat')
  arr = data['mat']
  mask = sampler.sample(arr, 0.3)
  dic = sampler.arrToLists(arr, mask)
  rows = sampler.toStr(range(arr.shape[0]))
  columns = sampler.toStr(range(arr.shape[1]))
  recon = recommend(dic, rows, columns)
  recon = sampler.listsToArr(recon, arr.shape)
  mse = npla.norm(recon - arr, 'fro')
  output_path = '../result/scaum/ratio_0_3/'
  os.system('mkdir -p ' + output_path)
  sio.savemat(output_path + 'result.mat', {'mat': arr, 'recon': recon, 'mse': mse})
