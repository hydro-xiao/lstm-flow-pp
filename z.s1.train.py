import sys
sys.path.append('./lstmflowprdc/')
import torch
import random
import os
import numpy as np
import xarray as xr
from lstmflowprdc import utils, normalize, core
from lstmflowprdc.train import TrainLSTM
from torch.utils.data import DataLoader

# Fix random seed
seedid = 111111
random.seed(seedid)
torch.manual_seed(seedid)
np.random.seed(seedid)
torch.cuda.manual_seed(seedid)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
################################################
################# Header Ends ##################
################################################


###### Read config file
config = utils.read_config('./main.config.test.23.crossvd')

###### Read static inputs
print('Reading static parameters...')
df_stc, n_stc_var = utils.read_stc_inputs(config)

###### Read dynamic inputs and natural flow (target)
print('Reading dynamic inputs for each basin...')
xr_dyn_rec, [n_dyn_var, n_stn] = utils.read_dyn_inputs(config)
flow_rec = utils.read_flow_obs(config)

###### For each station, divide flow with (area*mean_p)
for ni in range(n_stn):
    areai = df_stc['size'].iloc[ni]
    pavgi = df_stc['p_mean'].iloc[ni]
    flow_rec.loc[dict(id=ni)] = flow_rec.sel(id=ni) / pavgi

###### Normalize data, avg and std of the training data are
###### also applied in the verifcation and prediction process
print('Normalize the inputs based on stats of the training period...')
t_train = list(config['TEST_PARA']['Ttrain'].split(','))
in_tmp  = normalize.norm_dyn(xr_dyn_rec, config,  t_train,  t_train)
out_tmp = normalize.norm_dyn(flow_rec,   config,  t_train,  t_train)
stc_tmp = normalize.norm_stc(df_stc, config)
stc_epd = stc_tmp
#stc_epd = stc_tmp.expand_dims(dim={"time": in_tmp.sizes['time']}, axis=1)

#print( np.sum(np.sum(np.isnan(np.array(flow_rec) ) ) ) )

###### Convert data to pytorch Dataset Class
train_all = xr.merge([in_tmp, stc_epd, out_tmp])
target   = list(config['TEST_PARA']['target_var'].split(','))
features = [xs for xs in train_all.keys() if xs not in target]

train_dataset = core.seqDataset(
    train_all,
    target=target,
    features=features,
    seq_len=int(config['HYPER_PARA']['rho'])
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

###### Train model and save at selected epoch
nx = n_dyn_var + n_stc_var
ny = 1
nt = int(config['HYPER_PARA']['rho'])

lossfunc = core.RMSE_Loss()
main_model = core.LSTMmodel(nx=nx, ny=ny, \
        hiddensize=int(config['HYPER_PARA']['hidden_size']) )

print('Begin training the model...')
main_model = TrainLSTM(main_model, train_loader,\
         lossfunc, config)

