from configparser import ConfigParser, ExtendedInterpolation
import numpy  as np

def cal_rmse(pred, obs):
    err = np.array(pred) - np.array(obs)
    rmse = np.sqrt(np.mean(np.power(err,2)))
    return(rmse)


def cal_nse(pred, obs):
    errup = np.power(np.array(pred) - np.array(obs),2)
    errdn = np.power(np.array(obs)- np.mean(np.array(obs)), 2)
    nse = 1-np.sum(errup)/np.sum(errdn)
    return(nse)

def cal_kge(pred, obs):
    cc   = np.corrcoef(pred, obs)[0,1]
    pavg = np.mean(pred)
    pstd = np.std(pred)
    oavg = np.mean(obs)
    ostd = np.std(obs)
    kge  = 1-np.sqrt( np.power(1-cc,2)+ np.power(pavg/oavg-1,2)+\
            np.power(pstd/ostd-1,2) )
    return(kge)

