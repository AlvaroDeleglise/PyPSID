""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """
"Tools for simulating models"

import numpy as np
from scipy import stats

def addConjs(vals):
    vals = np.atleast_2d(vals).T
    valsConj = vals.conj()
    valsConj[np.abs(vals - valsConj) < np.spacing(1)] = np.nan
    return np.concatenate((vals, valsConj), axis=1)


def drawRandomPoles(N, poleDist = dict):
    nCplx = int(np.floor(N/2))
    
    # mag = np.random.rand(nCplx) # Uniform dist
    a, b = 2, 1

    # import matplotlib.pyplot as plt    
    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(0, 1, 100)
    # ax.plot(x, stats.beta.pdf(x, a, b),
    #         'r-', lw=5, alpha=0.6, label='beta pdf (a={}, b={})'.format(a, b))
    # ax.legend()
    # plt.show()

    mag = stats.beta.rvs(a=a, b=b, size=nCplx) # Beta dist

    theta = np.random.rand(nCplx) * np.pi
    vals = mag * np.exp( 1j * theta ) 

    valsA = addConjs(vals)
    valsA = valsA.reshape(valsA.size)
    valsA = valsA[ np.logical_not(np.isnan(valsA)) ]

    # Add real mode(s) if needed
    nReal = N-2*nCplx
    if nReal > 0:
        # rVals = np.random.rand(nReal)
        rVals = stats.beta.rvs(a=a, b=b, size=nReal) # Beta dist
        rSign = 2*(((np.random.rand(nReal) > 0.5).astype(float))-0.5)

        valsA = np.concatenate((valsA, rVals * rSign))

    return valsA 
    
