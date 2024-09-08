import ydata_profiling as ydp
import numpy as np
import pandas as pd

data1 = np.array([[0.116, 0.088, 0.117, 0.059, 0.071, 0.06, 0.046],
                 [0.227, 0.224, 0.245, 0.206, 0.215, 0.213, 0.188],
                 [0.174, 0.110, 0.171, 0.068, 0.081, 0.066, 0.055],
                 [0.340, 0.233, 0.430, 0.138, 0.178, 0.148, 0.125]])

data2 = np.array([[0.176, 0.130, 0.166, 0.153, 0.088, 0.107, 0.083, 0.099, 0.077, 0.092, 0.068, 0.066, 0.074, 0.062, 0.059],
                 [0.275, 0.231, 0.259, 0.252, 0.193, 0.216, 0.206, 0.211, 0.169, 0.214, 0.147, 0.147, 0.200, 0.198, 0.150],
                 [0.180, 0.113, 0.156, 0.181, 0.079, 0.103, 0.080, 0.096, 0.070, 0.091, 0.068, 0.060, 0.069, 0.060, 0.054],
                 [0.288, 0.121, 0.231, 0.167, 0.070, 0.108, 0.067, 0.092, 0.059, 0.079, 0.047, 0.044, 0.053, 0.043, 0.039],
                 [0.330, 0.319, 0.321, 0.296, 0.185, 0.186, 0.177, 0.171, 0.163, 0.165, 0.124, 0.121, 0.120, 0.120, 0.112],
                 [0.398, 0.113, 0.245, 0.288, 0.074, 0.120, 0.068, 0.156, 0.051, 0.082, 0.056, 0.044, 0.053, 0.043, 0.041]])

trans_data1 = data1.T
trans_data2 = data2.T

Frame_data1 = pd.DataFrame(data1, columns=['R','G', 'B', 'R+G', 'R+B', 'G+B', 'RGB'])
Frame_data2 = pd.DataFrame(data2, columns=['R','G', 'B', 'IR', 'R+G', 'R+B', 'G+B', 'R+IR', 'G+IR', 'B+IR', 'RGB', 'RG+IR', 'RB+IR', 'GB+IR', 'RGB+IR'])

Frame_trans1 = pd.DataFrame(trans_data1, columns=['PGD+Seg-RGB', 'PGD+DeepLabV3-RGB', 'MIFGSM+Seg-RGB', 'MIFGSM+DeepLabV3-RGB'])
Frame_trans2 = pd.DataFrame(trans_data2, columns=['PGD+Seg-MIX', 'PGD+DeepLabV3-MIX', 'PGD+MF', 'MIFGSM+Seg-MIX', 'MIFGSM+DeepLabV3-MIX', 'MIFGSM+MF'])

ydp.ProfileReport(Frame_data1).to_file('data1.html')
ydp.ProfileReport(Frame_data2).to_file('data2.html')
ydp.ProfileReport(Frame_trans1).to_file('trans1.html')
ydp.ProfileReport(Frame_trans2).to_file('trans2.html')
