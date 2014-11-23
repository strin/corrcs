import unittest
import numpy as np
import numpy.random as npr

import sampler
import score

class TestSampler(unittest.TestCase):

  def setUp(self):
    pass
  
  def test_sampler_findindex(self):
    arr = [1, 0, 1, 0, 1]
    condition = lambda x : x == 0
    self.assertEqual(sampler.findIndex(arr, condition), [1,3])

  def test_sampler_findindex2(self):
    arr = npr.randint(0, 1000, 100)
    condition = lambda x : x == x
    self.assertEqual(sampler.findIndex(arr, condition), range(len(arr)))

  def test_sampler_findindex3(self):
    arr = [float('NaN'), 1, 2, 3]
    condition = lambda x : x == x
    self.assertEqual(sampler.findIndex(arr, condition), [1,2,3])

  def test_sampler_findElement(self):
    arr = [3, 4, 5, 6, 2, 1]
    condition = lambda x : x >= 4 and x <= 7
    self.assertEqual(list(sampler.findElement(arr, condition)), [4,5,6])

  def test_sampler_sampleByRow(self):
    arr = [[1,2,3],[float('NaN'), 5, 6]]
    mask = sampler.sampleByRow(arr, 1)
    assert((mask == np.array([[1,1,1],[0,1,1]])).all())

class TestEval(unittest.TestCase):

  def test_mse(self):
    arr1 = [[1,2], [3,4]]
    arr2 = [[2,1], [3,6]]
    self.assertEqual(score.mse(arr1, arr2), 1.5)

  def test_mse2(self):
    arr1 = [[1,2], [3,4]]
    arr2 = [[2,1], [3, float('NaN')]]
    self.assertEqual(score.mse(arr1, arr2), 2.0/3)

if __name__ == "__main__":
  unittest.main()
    
