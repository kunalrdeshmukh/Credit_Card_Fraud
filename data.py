import sklearn as sk
import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt

Fraud_Data = np.genfromtxt('creditcard.csv',delimiter=',')
print ('Data imported')
Fraud_Data = np.delete(Fraud_Data,(0),axis=0)
# print len(Fraud_Data[:,1])
# print len(Fraud_Data[1])



