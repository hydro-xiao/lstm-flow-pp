import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from lstmflowprdc.calmetric import cal_rmse, cal_nse, cal_kge

def plt_mon(pred, qsim, obs, outf, lbs):
    rmseml = cal_rmse(pred, obs)
    nse = cal_nse(pred, obs)

    rmsesim = cal_rmse(qsim, obs)
    nsesim = cal_nse(qsim, obs)

    ccml   = np.corrcoef(pred, obs)[0,1]
    kgeml  = cal_kge(pred, obs)

    ccsim  = np.corrcoef(qsim, obs)[0,1]
    kgesim = cal_kge(qsim, obs)

    print(round(nse,3), round(kgeml,3), round(ccml,3), \
        round(nsesim,3), round(kgesim,3), round(ccsim,3), lbs)

    fig = plt.figure(figsize=(8.5,3.55))
    plt.plot(obs,  'k--', label='OBS, '+lbs)
    plt.plot(pred, 'r', label='LSTM (RSME='+"{:.2f}".format(rmseml)+\
            ', NSE='+"{:.2f}".format(nse) + ")")

    plt.plot(qsim, 'c', label='WRF-hyd (RSME='+"{:.2f}".format(rmsesim)+\
            ', NSE='+"{:.2f}".format(nsesim) + ")")
    plt.ylabel('Monthly FLow (Kaf)')
    plt.xlim(pd.Timestamp('2017-10-01'), pd.Timestamp('2020-09-30'))
    plt.xticks()
    plt.legend(framealpha=0)
    plt.savefig(outf, dpi=180)
    plt.close()

