import sys
sys.path.append('./lstmflowprdc/')
import torch
import random
import os
import numpy as np
import pandas as pd
import xarray as xr
from lstmflowprdc import utils, normalize, core
from lstmflowprdc.train import TrainLSTM
from lstmflowprdc.mkplts import plt_mon
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
df_stc_ori = df_stc.copy(deep=True)

###### Read dynamic inputs and natural flow (target)
print('Reading dynamic inputs for each basin...')
xr_dyn_rec, [n_dyn_var, n_stn] = utils.read_dyn_inputs(config)
flow_rec = utils.read_flow_obs(config)
xr_dyn_ori = xr_dyn_rec.copy(deep=True)

###### units adjustment constants
kaf_2_mm3 = 1.2335e15
tmpc1 = kaf_2_mm3 ### Kaf to mm^3
tmpc2 = 10 ** 6   ### m^2 to mm^2
tmpc3 = 1.

###### For each station, divide flow with (area*mean_p)
for ni in range(n_stn):
    areai = df_stc_ori['size'].iloc[ni]
    pavgi = df_stc_ori['p_mean'].iloc[ni]
    flow_rec.loc[dict(id=ni)] = flow_rec.sel(id=ni) / pavgi

###### Normalize data, avg and std of the training data are
###### also applied in the verifcation and prediction process
print('Normalize the inputs based on stats of the training period...')
t_train = list(config['TEST_PARA']['Ttrain'].split(','))
t_valid = list(config['TEST_PARA']['Tvalid'].split(','))
in_tmp  = normalize.norm_dyn(xr_dyn_rec, config,  t_train,  t_valid)
out_tmp = normalize.norm_dyn(flow_rec,   config,  t_train,  t_valid)
stc_tmp = normalize.norm_stc(df_stc, config)
stc_epd = stc_tmp.expand_dims(dim={"time": in_tmp.sizes['time']}, axis=1)

###### Convert data to pytorch Dataset Class
train_all = xr.merge([in_tmp, stc_epd, out_tmp])
target   = list(config['TEST_PARA']['target_var'].split(','))
features = [xs for xs in train_all.keys() if xs not in target]
print(target)
print(features)

train_dataset = core.seqDataset(
    train_all,
    target=target,
    features=features,
    seq_len=12
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


###### Train model and save at selected epoch
nx = n_dyn_var + n_stc_var
ny = 1
nt = int(config['HYPER_PARA']['rho'])

lossfunc = core.RMSE_Loss()

nams = pd.read_csv( config['INPUT']['basin_listf'] )
saveFolder  =  config['INPUT']['savemodel_dir']
outputdir   =  config['INPUT']['output_dir']
figFolder   =  config['INPUT']['figure_dir']
os.system(f'mkdir -p {outputdir}')
os.system(f'mkdir -p {figFolder}')

epoch_use   = config['HYPER_PARA']['EPOCH_use']
mk_figs     = config['INPUT']['make_plot']

if 1:
        modelFile = os.path.join(
            saveFolder, "model_Ep" + str(epoch_use) + ".pt"
            )
        model = torch.load(modelFile)
        loss_ttl = 0.
        ####### main loop over data
        outrec = []
        for (batch_idx, batch) in enumerate(train_loader):
            xi = batch[0][0,:,:,:]
            yi = batch[1][0,:,:,:]
            model_out  = model(xi)
            outrec.append(normalize.trans_to_flow(model_out, flow_rec, config, t_train) )

y_predict = np.concatenate((outrec[:]), axis=0)

###### flow mulitply by area and pavg
for ni in range(n_stn):
    areai = df_stc_ori['size'].iloc[ni]
    pavgi = df_stc_ori['p_mean'].iloc[ni]
    flow_rec.loc[dict(id=ni)] = flow_rec.sel(id=ni)* (areai*tmpc2) * pavgi/tmpc1
    y_predict[:,ni] = y_predict[:,ni] * (areai*tmpc2)*pavgi/tmpc1
    xr_dyn_ori['Qsim'].loc[dict(id=ni)] =  xr_dyn_ori['Qsim'].sel(id=ni)* (areai*tmpc2) / tmpc1

obs = flow_rec.sel(time=slice(t_valid[0], t_valid[1]))
qsim = xr_dyn_ori['Qsim'].sel(time=slice(t_valid[0], t_valid[1]))

###### convert daily flow to monthly total
indx_use  = pd.date_range(t_valid[0], t_valid[1], freq='M')
for bi in range(n_stn):
    y_predict[:,bi]       = y_predict[:,bi]*indx_use.days_in_month
    obs.loc[dict(id=bi)]  = obs.isel(id=bi)*indx_use.days_in_month
    qsim.loc[dict(id=bi)] = qsim.isel(id=bi)*indx_use.days_in_month

###### save output to csv files and make figures
for bi in range(n_stn):
    predbi       = pd.Series(y_predict[:,bi])
    predbi.index = indx_use

    predbi.to_csv(outputdir+'/'+str(bi+1)+'.predicted-flow.'+\
            t_valid[0]+'-'+t_valid[1]+'.csv',\
            index_label='date', header=['flow'])

    if int(mk_figs):
    	obsbi        = pd.Series(obs.isel(id=bi).to_array()[0,:])
    	obsbi.index  = pd.date_range(t_valid[0], t_valid[1], freq='M')

    	qsimbi       = pd.Series(qsim.isel(id=bi)[:])
    	qsimbi.index = pd.date_range(t_valid[0], t_valid[1], freq='M')

    	plt_mon(predbi, qsimbi, obsbi, figFolder+'/ep'+str(epoch_use)+\
            '.'+str(bi+1)+'.png', \
            nams['fname'].iloc[bi]+' ('+nams['id'].iloc[bi]+')')
if int(mk_figs):
    print('The cols above are: lstm-nse, lstm-kge, lstm-cc, qsim-nse, qsim-kge, qsim-cc')

