import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ydata_profiling as ydp


data_ch3 = np.asarray([
  [0.116, 0.088, 0.117, 0.059, 0.071, 0.060, 0.046],
  [0.227, 0.224, 0.245, 0.206, 0.215, 0.213, 0.188],
  [0.174, 0.110, 0.171, 0.068, 0.081, 0.066, 0.055],
  [0.340, 0.233, 0.430, 0.138, 0.178, 0.148, 0.125],
])
data_ch3_cols = ['R', 'G', 'B', 'R+G', 'R+B', 'G+B', 'RGB']
data_ch3_rows = ['PGD+Seg', 'PGD+DV3', 'MIFGSM+Seg', 'MIFGSM+DV3']
data_ch4 = np.asarray([
  [0.176, 0.130, 0.166, 0.153, 0.088, 0.107, 0.083, 0.099, 0.077, 0.092, 0.068, 0.066, 0.074, 0.062, 0.059],
  [0.275, 0.231, 0.259, 0.252, 0.193, 0.216, 0.206, 0.211, 0.169, 0.214, 0.147, 0.147, 0.200, 0.198, 0.150],
  [0.180, 0.113, 0.156, 0.181, 0.079, 0.103, 0.080, 0.096, 0.070, 0.091, 0.068, 0.060, 0.069, 0.060, 0.054],
  [0.288, 0.121, 0.231, 0.167, 0.070, 0.108, 0.067, 0.092, 0.059, 0.079, 0.047, 0.044, 0.053, 0.043, 0.039],
  [0.330, 0.319, 0.321, 0.296, 0.185, 0.186, 0.177, 0.171, 0.163, 0.165, 0.124, 0.121, 0.120, 0.120, 0.112],
  [0.398, 0.113, 0.245, 0.288, 0.074, 0.120, 0.068, 0.156, 0.051, 0.082, 0.056, 0.044, 0.053, 0.043, 0.041],
])
data_ch4_cols = ['R', 'G', 'B', 'IR', 'R+G', 'R+B', 'G+B', 'R+IR', 'G+IR', 'B+IR', 'RGB', 'RG+IR', 'RB+IR', 'GB+IR', 'RGB+IR']
data_ch4_rows = ['PGD+Seg', 'PGD+DV3', 'PGD+MF', 'MI+Seg', 'MI+DV3', 'MI+MF']

ydp.ProfileReport(pd.DataFrame(data_ch3,   columns=data_ch3_cols)).to_file('./img/ch3.html')
ydp.ProfileReport(pd.DataFrame(data_ch3.T, columns=data_ch3_rows)).to_file('./img/ch3_T.html')
ydp.ProfileReport(pd.DataFrame(data_ch4,   columns=data_ch4_cols)).to_file('./img/ch4.html')
ydp.ProfileReport(pd.DataFrame(data_ch4.T, columns=data_ch4_rows)).to_file('./img/ch4_T.html')
