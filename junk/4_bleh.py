# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 00:30:12 2017

@author: mohamed
"""

import numpy as np
import matplotlib.pylab as plt

arr = np.array([[0, 0.7], [1, 0.72], [2, 0.74], [3, 0.8], [4, 0.81], [5, 0.82], [6, 0.825], [7, 0.83], [8, 0.831]])
title = "haha"
xlab = "..."
ylab = "..."
savename = "/home/mohamed/Desktop/bleh.svg"
arr2=None
vline=None
hline1=0.7
hline2=None
IS_CI=True

print("Plotting " + title)
        
#fig, ax = plt.subplots() 
#plt.subplots() 
plt.figure(figsize=(5, 5))

plt.plot(arr[:,0], arr[:,1], color='b', linewidth=2.5, aa=False)
if arr2 is not None:
    plt.plot(arr[:,0], arr2, color='r', linewidth=2.5, aa=False)
if vline is not None:
    #plt.axvline(x=vline, linewidth=1.5, color='k', linestyle='--')
    pass
if hline1 is not None:
    plt.axhline(y=hline1, linewidth=1.5, color='b', linestyle='--')
if hline2 is not None:
    plt.axhline(y=hline2, linewidth=1.5, color='r', linestyle='--')
if IS_CI:
    #plt.ylim(ymax=1)
    plt.ylim(ymin=0.5, ymax=1)
    #plt.axhline(y=0.5, linewidth=1.5, color='k', linestyle='--')

plt.title(title, fontsize =16, fontweight ='bold')
plt.xlabel(xlab)
plt.ylabel(ylab)       
plt.tight_layout()        

#ax.axis('equal')
#ax.set_aspect('box')
#ax.set_aspect('equal')        
#plt.figaspect(1.0)
#plt.tight_layout()        
#ax.imshow(ax, aspect='auto') 
#ax.set_aspect(1.0)

plt.savefig(savename)
plt.close()
