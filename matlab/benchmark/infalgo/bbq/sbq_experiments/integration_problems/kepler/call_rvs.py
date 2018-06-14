import numpy as np
import orbit_rvonly as o
import sys
import os

mayor = 'RVs_Mayor2009'
vogt = 'RVs_Vogt2010'

def ReadData():
    root = os.path.dirname(__file__)
    X = np.genfromtxt('%s/%s' % (root, mayor)).T
    jd_m, rv_m, erv_m = X[0], X[1] * 1e3, X[2] * 1e3
    X = np.genfromtxt('%s/%s' % (root, vogt)).T
    jd_v, rv_v, erv_v = X[0] + 2450000, X[1], X[2]
    t0 = np.floor(min(jd_m.min(), jd_v.min()))
    n_m, n_v = len(jd_m), len(jd_v)
    x = np.zeros((2, n_m + n_v))
    x[0,:n_m], x[0,n_m:] = jd_m - t0, jd_v - t0
    x[1,n_m:] = 1
    y = np.zeros((2, n_m + n_v))
    rvm_m, rvm_v = rv_m.mean(), rv_v.mean()
    y[0,:n_m], y[0,n_m:] = rv_m - rvm_m, rv_v - rvm_v
    y[1,:n_m], y[1,n_m:] = erv_m**2, erv_v**2
    return x, y, (t0, rvm_m, rvm_v)

def ModelFunc(par, x, y):
    n_inst = len(np.unique(x[1,:]))
    m = np.zeros(len(y[0,:]))
    for i in range(n_inst):
        m[x[1,:] == i] = par[n_inst + i]
    n_pl = (len(par) - 2 * n_inst) / 5
    for i in range(n_pl):
        ii = 2 * n_inst + 5 * i
        m += o.radvel(x[0,:], par[ii], par[ii + 1], par[ii + 2], \
                      0.0, par[ii + 3], par[ii + 4])
    return m

def ErrFunc(par, x, y):
    n_inst = len(np.unique(x[1,:]))
    err2 = y[1,:]
    for i in range(n_inst):
        err2[x[1,:] == i] += par[i]
    return (y[0,:] - ModelFunc(par, x, y))**2 / err2

def NegLogLike(par, x, y):
    return np.sum(ErrFunc(par, x, y))
    
if __name__ == "__main__":
    x, y, ref = ReadData()
    par = [float(z) for z in sys.argv[1:]]
    print NegLogLike(par, x, y)
