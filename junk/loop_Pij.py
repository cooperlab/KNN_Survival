# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:43:31 2017

@author: mohamed
"""

import numpy as np
import tensorflow as tf

#%%

n = 10; d = 30
#x_transformed = np.random.rand(n, d)
#
#
#normax = np.zeros((n, n))
#
##sid = 0
#
#for sid in range(n):
#    patient = x_transformed[sid, :]
#    patient_normax = (patient[None, :] - x_transformed)**2
#    normax[:,sid] = np.sum(patient_normax, axis=1)

#%%

tf.reset_default_graph()

AX = tf.random_normal((n, d))

# first patient
sID = 0
patient = AX[sID, :]
patient_normax = (patient[None, :] - AX)**2
normAX = tf.reduce_sum(patient_normax, axis=1)
normAX = normAX[None, :]

# all other patients
def _append_normAX(sID, normAX):

    """append normAX for a single patient to existing normAX"""
    
    # calulate normAX for this patient    
    patient = AX[sID, :]
    patient_normax = (patient[None, :] - AX)**2
    patient_normax = tf.reduce_sum(patient_normax, axis=1)

    # append to existing list
    normAX = tf.concat((normAX, patient_normax[None, :]), axis=0)
    
    # sID++
    sID = tf.cast(tf.add(sID, 1), tf.int32)
    
    return sID, normAX
    

# Go through all patients and add normAX
sID = tf.cast(tf.Variable(0), tf.int32)

c = lambda sID, normAX: tf.less(sID, tf.cast(n, tf.int32))
b = lambda sID, normAX: _append_normAX(sID, normAX)

normAX = tf.while_loop(c, b, [sID, normAX])