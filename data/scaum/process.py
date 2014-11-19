'''
extract from polling data "scaum-611.csv" statistical averages. 
* take all features (column) with no data missing.
* for every *gap* years, compute averages of feature.
* output a matrix of *period* x *feature*, save it to scaum-611.mat
'''
import os
import numpy as np
import scipy.io as sio

class signal:
  def __init__(self, gap):
    self.gap = gap
    self.mean = None
    self.year = None
    self.data = None

data = np.genfromtxt('scaum-611.csv', delimiter=',')
_isnan = np.mean(np.isnan(data[1:]), axis=0)
_nonzero = [i for (i, x) in enumerate(_isnan) if x == 0 and i != 0]
print _nonzero
gap = 10
first_year = data[0][0]
sig = list()
_year = list()
_data = list()
mat = list()
for (ri, row) in enumerate(data):
  if ri == 0:
    continue
  _year.append(int(row[0]))
  row = row[_nonzero]
  s = signal(gap)
  _data.append(row)
  if ri % gap == 0:
    s.year = _year
    _year = list()
    _mdata = np.ma.masked_array(_data, np.isnan(_data))
    s.mean = np.mean(_mdata, axis=0)
    _data = list()
    sig.append(s)
    mat.append(s.mean)

print len(sig)
print 'example year = ', sig[0].year
print 'example mean = ', sig[0].mean

sio.savemat('scaum-611', {'mat': mat})
