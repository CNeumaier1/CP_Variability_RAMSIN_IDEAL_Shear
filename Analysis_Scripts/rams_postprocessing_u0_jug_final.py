import xarray as xr
import numpy as np
import pandas as pd
import glob
import h5py
import os
import matplotlib.pyplot as plt
from jug import TaskGenerator


def get_var(filePath, variables):
    #This function gets a variable/variables from the given file using xarray

    if os.path.exists(filePath):

        
        ds = xr.open_dataset(filePath, phony_dims = 'sort')[variables]
        if len(ds.dims)==3:
            if 'phony_dim_0' not in ds.dims:
                ds = ds.rename_dims({'phony_dim_3':'z', 'phony_dim_2':'y', 'phony_dim_1':'x'})
                ds = ds.assign_coords(z=ds.z)
            else:    
                if ny == 3000:
                    ds = ds.rename_dims({'phony_dim_0':'z', 'phony_dim_1':'y', 'phony_dim_2':'x'})
                    ds = ds.assign_coords(z=ds.z)
                else:
                    ds = ds.rename_dims({'phony_dim_2':'z', 'phony_dim_1':'x', 'phony_dim_0':'y'})
                    ds = ds.assign_coords(z=ds.z)
        elif len(ds.dims)==2:
            ds = ds.rename_dims({'phony_dim_0':'y', 'phony_dim_1':'x'})
        elif len(ds.dims)==4:
            if 'phony_dim_5' in ds.dims:
                ds = ds.rename_dims({'phony_dim_3':'np','phony_dim_5':'nzg', 'phony_dim_0':'y','phony_dim_1':'x'})
                ds = ds.assign_coords(nzg=ds.nzg)
        
        
        ds = ds.assign_coords(y=ds.y)
        ds = ds.assign_coords(x=ds.x)

        #change the points to be dropped based on your grid size
        ds = ds.drop_isel(x=[0,nx-1],y=[0,ny-1])

        return(ds)
    else:
        print(f'Error: file {filePath} does not exist.')

@TaskGenerator
def get_rams_thv(wind_prof,CP1_T, CP2_T, time_d, savePath):
    '''gets thv for rams_case'''
    # print(f"CP1 {CP1_T}K CP1 {CP1_T}K {time_d} min
    base_Path = f"/moonbow/cneumaie/CP_train_derecho/"
    CP1_paths_u0 = [f"{base_Path}CP1_{i}k_u0_derecho/" for i in [5,10,20]]
    CP1_paths_shear = [f"{base_Path}CP1_{i}k_derecho/" for i in [5,10,20]]
    CP2_20k_paths_u0 = [f"{base_Path}CP1_{j}k_CP2_20k_u0_background_tracer_{i}min_derecho/" for i in [30,60,90,120] for j in [5,10,20]]
    CP2_10k_paths_u0 = [f"{base_Path}CP1_{j}k_CP2_10k_u0_background_tracer_{i}min_derecho/" for i in [30,60,90,120] for j in [5,10,20]]
    CP2_5k_paths_u0 = [f"{base_Path}CP1_{j}k_CP2_5k_u0_background_tracer_{i}min_derecho/" for i in [30,60,90,120] for j in [5,10,20]]
    
    CP1_keys_u0 = [f"CP1 {i}K" for i in [5,10,20]]
    CP1_dict_u0 = dict(zip(CP1_keys_u0,CP1_paths_u0))
    CP2_20k_keys_u0 = [f"CP1 {j}K {i} min" for i in [30,60,90,120] for j in [5,10,20]]
    CP2_20k_dict_u0 = dict(zip(CP2_20k_keys_u0,CP2_20k_paths_u0)) 
    CP2_10k_keys_u0 = [f"CP1 {j}K {i} min" for i in [30,60,90,120] for j in [5,10,20]]
    CP2_10k_dict_u0 = dict(zip(CP2_10k_keys_u0,CP2_10k_paths_u0)) 
    CP2_5k_keys_u0 = [f"CP1 {j}K {i} min" for i in [30,60,90,120] for j in [5,10,20]]
    CP2_5k_dict_u0 = dict(zip(CP2_5k_keys_u0,CP2_5k_paths_u0)) 

    
    for wind in ['placeholder', 'u0']:
       
        if wind == 'u0':
            for CP1_temp in [5,10,20]:
                CP1_key = f"CP1 {CP1_temp}K"
                CP1_list = sorted(glob.glob(CP1_dict_u0[CP1_key]+'a-L'+'*.h5'))
                for CP2_temp in [5,10,20]:
                    for time in [30, 60, 90,120]:
                        CP2_key = f"CP1 {CP1_temp}K {time} min"
                        if time ==30:
                            file_list = [CP1_list[:6]]
                        elif time ==60:
                            file_list = [CP1_list[:12]]
                        elif time ==90:
                            file_list = [CP1_list[:18]]
                        elif time ==120:
                            file_list = [CP1_list[:24]]
                        if CP2_temp == 20:
                            file_list.extend(sorted(glob.glob(CP2_20k_dict_u0[CP2_key]+'a-L'+'*.h5')))
                        elif CP2_temp == 10:
                            file_list.extend(sorted(glob.glob(CP2_10k_dict_u0[CP2_key]+'a-L'+'*.h5')))
                        elif CP2_temp == 5:
                            file_list.extend(sorted(glob.glob(CP2_5k_dict_u0[CP2_key]+'a-L'+'*.h5')))
                        file_list_combined = file_list[0]+file_list[1:]
                        times = np.arange(len(file_list_combined))
                        if CP1_key == f"CP1 {CP1_T}K" and CP2_key ==f"CP1 {CP1_T}K {time_d} min" and CP2_temp ==CP2_T and wind == wind_prof: 
                            times_list = []
                            for t in times:
                                # print(t)
                                ds_cp = get_var(file_list_combined[t], ['THETA', 'RV', 'RTP', 'PI', 'UC','VC','WC','TRACERP001'])
                                ds_cp_mean = ds_cp.isel(z=slice(0,99)).mean(dim= 'x')
                                times_list.append(ds_cp_mean)
                                ds_cp.close()
                            # print('done')
                            time_array = xr.concat(times_list, dim = "t")
    time_array.to_netcdf(savePath)
    time_array.close()

dataPath_100m = '/moonbow/cneumaie/RAMS_output/CP1_tracer/'

z_test = []
header = sorted(glob.glob(dataPath_100m + '*.txt'))[0]
f = open(header)
lines = f.readlines()
for i in range(len(lines)):
    if lines[i] == '__ztn01\n':
        for j in range(2,162):
            z_test.append(round(float(lines[i+j].strip('\n').strip(' ')),0))
z_levels = np.array([int(i) for i in z_test])
alt = z_levels
nx = 250
ny = 3000
dx = dy = 100

savePath = f"/moonbow/cneumaie/CP_train_derecho/processed/"
# wind_prof = "u0"
# CP1_T = CP2_T = 20
# timedelta = 120
CP1_Tlist = [5,10,20]
CP2_Tlist = [5,10,20]
time_list = [30,60,90,120]
wind_list = ['u0','placeholder']

for wind_prof in wind_list:
    for CP1_T in CP1_Tlist:
        for CP2_T in CP2_Tlist:
            for timedelta in time_list:

                filename = f"{savePath}CP1_{CP1_T}_CP2_{CP2_T}_{timedelta:03}_{wind_prof}.nc"
                get_rams_thv(wind_prof,CP1_T, CP2_T, timedelta, filename)
