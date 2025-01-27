import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
from PIL import Image
import glob
import h5py
from scipy.ndimage import binary_erosion, label
from matplotlib.lines import Line2D
from jug import TaskGenerator
import jug

def calc_theta_rho(ds, count):
    '''Calculates density potential temperature'''
    if count == 0:
        ds2 = ds.assign(theta_rho = ds.THP *((1-ds.RTP)+(1.61*ds.RV) ))
    else:
        ds2 = ds.assign(theta_rho = ds.THETA *((1-ds.RTP)+(1.61*ds.RV) ))
    
    return ds2
def calc_cp_prime(outfilename, cp2_start, thv_mean):
    '''calculates cp_prime for cp 1 and cp2'''
    test_dataset = h5py.File(outfilename, 'r')
    cp_edges_n= np.array(test_dataset['cp_edges_n'])
    cp_edges_s= np.array(test_dataset['cp_edges_s'])
    lower_box_n= np.array(test_dataset['lower_box_n'])
    lower_box_s= np.array(test_dataset['lower_box_s'])
    base1_edge_array_n= np.array(test_dataset['base1_edge_array_n'])
    base1_edge_array_s= np.array(test_dataset['base1_edge_array_s'])
    cp2_min_ind_n = np.array(test_dataset['cp2_min_ind_n'])
    cp2_min_ind_s = np.array(test_dataset['cp2_min_ind_s'])
    test_dataset.close()
    
    thv_bar1 = np.zeros((len(thv_mean), len(thv_mean[0,:,0])))
    nt = len(thv_bar1)
    for t in range(nt):
        base1edge = int(base1_edge_array_n[t])
        base1_edge_s = int(base1_edge_array_s[t])
        if base1edge > base1_edge_s:
            thv_bar1[t,:] = np.mean(thv_mean[t,:,base1_edge_s:base1edge], axis = 1)  ###calculates environment for cp1
        else:
            thv_bar1[t,:] = np.mean(np.concatenate((thv_mean[t,:,base1_edge_s:],thv_mean[t,:,:base1edge]), axis=1), axis = 1)
    thv_prime_1 = thv_mean-thv_bar1[:,:,None]
    thv_bar2 = np.zeros((len(thv_mean), len(thv_mean[0,:,0])))
    cp1_edges = cp_edges_n[:,0]
    lower_box = lower_box_n
    for t in range(cp2_start,nt):
        if t == cp2_start or t == cp2_start+1:
            if "u0" in outfilename:
                base_state2 = np.mean((thv_mean[cp2_start-2,:,int(cp2_min_ind_n[cp2_start+1])], (thv_mean[cp2_start-2,:,int(cp2_min_ind_s[cp2_start+1])])), axis = 0)
            elif "shear" in outfilename:
                base_state2 = thv_mean[cp2_start-2,:,int(cp2_min_ind_n[cp2_start+1])]
                
            thv_bar2[t,:] = base_state2  ##calculates base state from before cold pool 2 for these 2 timesteps
        else:
            if "u0" in outfilename:
                if (np.isnan(cp2_min_ind_n[t])==False) and (np.isnan(cp2_min_ind_s[t])==False):
                    thv_bar2[t,:] = np.mean((thv_mean[t,:,int(cp2_min_ind_n[t])], (thv_mean[t,:,int(cp2_min_ind_s[t])])), axis = 0)
                else:
                    thv_bar2[t,:] = thv_bar1[t,:]
                    
            elif "shear" in outfilename:
                if (np.isnan(cp2_min_ind_n[t])==False):
                    thv_bar2[t,:] = thv_mean[t,:,int(cp2_min_ind_n[t])]
                else:
                    thv_bar2[t,:] = thv_bar1[t,:]
                

    thv_prime_2 = thv_mean[:,:,:]-thv_bar2[:,:,None]
    
    return thv_prime_1, thv_prime_2,thv_bar1,thv_bar2

def cp_depth(outfilename,thv_prime_mask,nt,thv_threshold,cp2_start,alt_list,q,cp_mode = 1):
    '''calulates a list of (max) depths for each time'''
    if cp_mode ==1:
        time_range = np.arange(2,cp2_start-2)
        cp_index = 0
    elif cp_mode ==2:
        time_range = np.arange(cp2_start,nt)
        cp_index = 1
        
    test_dataset = h5py.File(outfilename, 'r')
    cp_edges_n= np.array(test_dataset['cp_edges_n'])
    cp_edges_s= np.array(test_dataset['cp_edges_s'])   
    test_dataset.close()
    
    thv_prime_mask_c = np.nan_to_num(thv_prime_mask, nan=False)
    thv_prime_mask_copy = thv_prime_mask_c.astype(bool)
    max_depth_list = []

    for t in time_range:
            ### First need to find out if there are surface cold pool edgesm if not then we cannot calculate the edge depths
            if np.isnan(cp_edges_s[t,cp_index]) or np.isnan(cp_edges_n[t,cp_index]):
                max_depth_list.append(np.nan)
                continue
            elif np.isnan(cp_edges_s[t,cp_index]) and np.isnan(cp_edges_n[t,cp_index]):
                max_depth_list.append(np.nan)
                continue
            else:
                cp_edge_s = int(cp_edges_s[t,cp_index])
                cp_edge_n = int(cp_edges_n[t,cp_index])
        
            
            labeled_mask, num_features = label(thv_prime_mask_copy[t,:,cp_edge_s-20:cp_edge_n+10], structure = np.ones((3,3)))   ###seperate this out to be 2 masks, one for n edge and one for s edge

            # Find the lower region label
            # Assuming the lower region is the one that touches the bottom (lowest indices) in the z-dimension
            min_z = 0
            
            bottom_labels = np.unique(labeled_mask[min_z])
            bottom_labels = bottom_labels[bottom_labels != 0]  # Remove the background label (0)
            
            # Extract the lower region mask
            lower_region_mask = np.isin(labeled_mask, bottom_labels)
            # Using astype
            lower_region_mask = lower_region_mask.astype(int)
            border_indices_test = np.array(np.nonzero(lower_region_mask))


            # Extract y and z rows
            y_points = border_indices_test[1,:]
            z_points = border_indices_test[0,:]

            # Create a dictionary to store the maximum z for each unique y point
            max_z_for_y = {}
            for y, z in zip(y_points, z_points):
                if y in max_z_for_y:
                    max_z_for_y[y] = max(max_z_for_y[y], z)
                else:
                    max_z_for_y[y] = z

            # Convert dictionary to a list of tuples for easier reading, if needed
            max_z_for_y = list(max_z_for_y.values())
            max_z_north = max_z_for_y[-40:]
            max_z_south = max_z_for_y[:40]
            try:
                # Find the altitude point assosciated with each z index in each mask
                border_alt_n = [alt_list[int(k)] for k in max_z_north]
                border_alt_s = [alt_list[int(k)] for k in max_z_south]
                #Find the depth that is the q quantile of all the depths in each array
                highest_y_index_n = np.percentile(border_alt_n,q)
                highest_y_index_s = np.percentile(border_alt_s,q)

                if "u0" in outfilename:
                    max_depth_list.append(np.mean((highest_y_index_n,highest_y_index_s)))
                elif "shear" in outfilename:
                    max_depth_list.append(highest_y_index_n)
            except:
                max_depth_list.append(np.nan)
                continue
                
        
    return max_depth_list

# @TaskGenerator
def get_envir_winds(outfilename, cp2_start, u_mean, v_mean):
    '''calculates environmental wind profiles for cp1 and cp2'''
    
    test_dataset = h5py.File(outfilename, 'r')
    cp_edges_n= np.array(test_dataset['cp_edges_n'])
    cp_edges_s= np.array(test_dataset['cp_edges_s'])
    lower_box_n= np.array(test_dataset['lower_box_n'])
    lower_box_s= np.array(test_dataset['lower_box_s'])
    base1_edge_array_n= np.array(test_dataset['base1_edge_array_n'])
    base1_edge_array_s= np.array(test_dataset['base1_edge_array_s'])
    cp2_min_ind_n = np.array(test_dataset['cp2_min_ind_n'])
    cp2_min_ind_s = np.array(test_dataset['cp2_min_ind_s'])
    test_dataset.close()
    u_bar1 = np.zeros((len(u_mean), len(u_mean[0,:,0])))
    v_bar1 = np.zeros((len(u_mean), len(u_mean[0,:,0])))
    nt = len(u_bar1)
    for t in range(nt):
        base1edge = int(base1_edge_array_n[t])
        base1_edge_s = int(base1_edge_array_s[t])
        if base1edge > base1_edge_s:
            u_bar1[t,:] = np.mean(u_mean[t,:,base1_edge_s:base1edge], axis = 1)
            v_bar1[t,:] = np.mean(v_mean[t,:,base1_edge_s:base1edge], axis = 1)
        else:
            u_bar1[t,:] = np.mean(np.concatenate((u_mean[t,:,base1_edge_s:],u_mean[t,:,:base1edge]), axis=1), axis = 1)
            v_bar1[t,:] = np.mean(np.concatenate((v_mean[t,:,base1_edge_s:],v_mean[t,:,:base1edge]), axis=1), axis = 1)
    u_prime_1 = u_mean-u_bar1[:,:,None]
    u_bar2 = np.zeros((len(u_mean), len(u_mean[0,:,0])))
    v_prime_1 = v_mean-v_bar1[:,:,None]
    v_bar2 = np.zeros((len(u_mean), len(u_mean[0,:,0])))
    cp1_edges = cp_edges_n[:,0]
    lower_box = lower_box_n
    for t in range(cp2_start,nt):
        if t == cp2_start or t == cp2_start+1:
            if "u0" in outfilename:
                u_base_state2 = np.mean((u_mean[cp2_start-2,:,int(cp2_min_ind_n[cp2_start+1])], (u_mean[cp2_start-2,:,int(cp2_min_ind_s[cp2_start+1])])), axis = 0)
                v_base_state2 = np.mean((v_mean[cp2_start-2,:,int(cp2_min_ind_n[cp2_start+1])], (v_mean[cp2_start-2,:,int(cp2_min_ind_s[cp2_start+1])])), axis = 0)
            elif "shear" in outfilename:
                u_base_state2 = u_mean[cp2_start-2,:,int(cp2_min_ind_n[cp2_start+1])]
                v_base_state2 = v_mean[cp2_start-2,:,int(cp2_min_ind_n[cp2_start+1])]
                
            u_bar2[t,:] = u_base_state2  ##calculates base state from before cold pool 2 for these 2 timesteps
            v_bar2[t,:] = v_base_state2
        else:
  
            if "u0" in outfilename:
                if (np.isnan(cp2_min_ind_n[t])==False) and (np.isnan(cp2_min_ind_s[t])==False):
                    u_bar2[t,:] = np.mean((u_mean[t,:,int(cp2_min_ind_n[t])], (u_mean[t,:,int(cp2_min_ind_s[t])])), axis = 0)
                    v_bar2[t,:] = np.mean((v_mean[t,:,int(cp2_min_ind_n[t])], (v_mean[t,:,int(cp2_min_ind_s[t])])), axis = 0)
                else:
                    u_bar2[t,:] = u_bar1[t,:]
                    v_bar2[t,:] = v_bar1[t,:]
                    
            elif "shear" in outfilename:
                if (np.isnan(cp2_min_ind_n[t])==False):
                    u_bar2[t,:] = u_mean[t,:,int(cp2_min_ind_n[t])]
                    v_bar2[t,:] = v_mean[t,:,int(cp2_min_ind_n[t])]
                else:
                    u_bar2[t,:] = u_bar1[t,:]
                    v_bar2[t,:] = v_bar1[t,:]
                

    u_prime_2 = u_mean[:,:,:]-u_bar2[:,:,None]
    v_prime_2 = v_mean[:,:,:]-v_bar2[:,:,None]
    
    return u_prime_1, u_prime_2,u_bar1,u_bar2, v_prime_1, v_prime_2,v_bar1,v_bar2
                        
                        

def calc_depth(filePath,wind_prof,CP1_T, CP2_T, time_d,alt_list, thv, q):
    '''generates cp1 and cp2 depth arrays'''
    outfilename = filePath
    outfilename2 = f"./cpfile_edges_v7/cpfile_CP1_{CP1_T}k_CP2_{CP2_T}k_{time_d:03}min_{wind_prof}_v2.h5"
    outfilename3 = f"CP1_{CP1_T}k_CP2_{CP2_T}k_{time_d:03}min_{wind_prof}"
    if "30min" in outfilename2:
        cp2_start = 8
        plotting_cp_mode = 0
    elif "60min" in outfilename2:
        cp2_start = 14
        plotting_cp_mode = 1
    elif "90min" in outfilename2:
        cp2_start = 20
        plotting_cp_mode = 1
    elif "120min" in outfilename2:
        cp2_start = 26
        plotting_cp_mode = 0
    
    rams_data = xr.open_dataset(outfilename)
    rams_data2 = calc_theta_rho(rams_data, 1)
    thv_mean = rams_data2["theta_rho"]
    u_mean = rams_data2["UC"]
    v_mean = rams_data2["VC"]
    nt = len(thv_mean)
    thv_prime_cp1, thv_prime_cp2, thv_bar1,thv_bar2 = calc_cp_prime(outfilename2,cp2_start, thv_mean)
    u_prime_1, u_prime_2,u_bar1,u_bar2, v_prime_1, v_prime_2,v_bar1,v_bar2 = get_envir_winds(outfilename2,cp2_start,u_mean,v_mean)
    thv_threshold1 = -thv 
    if "30min" in outfilename:
        thv_threshold2 = -thv
    else:
        thv_threshold2 = -thv 
        
    thv_prime_mask_cp1 = np.where(thv_prime_cp1 < thv_threshold1, thv_prime_cp1, np.nan)
    thv_prime_mask_cp2 = np.where(thv_prime_cp2 < thv_threshold2, thv_prime_cp2, np.nan)
    
    max_depth_list_cp1 = cp_depth(outfilename2,thv_prime_mask_cp1,nt,thv_threshold1,cp2_start,alt_list,q,cp_mode = 1)
    max_depth_list_cp2 = cp_depth(outfilename2,thv_prime_mask_cp2,nt,thv_threshold2,cp2_start,alt_list,q,cp_mode = 2)

    
    return max_depth_list_cp1, max_depth_list_cp2, thv_bar1,thv_bar2, u_bar1,u_bar2,v_bar1,v_bar2

def make_cpdepth_array(filePaths,alt_list, thv, q):
    '''calculates cp depth and puts them into a dictionary and an array'''
    temp_list = [5,10,20]
    time_list = [30,60,90, 120]
    wind_list = ['shear', 'u0']
    
    speed_dict_cp1 = {}
    speed_dict_cp2 = {}
    wind_list_pd  = []
    cp1_temp_list = []
    cp2_temp_list = []
    
    time_list_pd = []
    cp1_depth_list_all = []
    cp2_depth_list_all = []

    test_array = np.zeros((len(wind_list),len(temp_list),len(temp_list),len(time_list),2,50))
    thv_array = np.zeros((len(wind_list),len(temp_list),len(temp_list),len(time_list),2,50,99))
    u_array = np.zeros((len(wind_list),len(temp_list),len(temp_list),len(time_list),2,50,99))
    v_array = np.zeros((len(wind_list),len(temp_list),len(temp_list),len(time_list),2,50,99))
    ###setting the time to start tracking the edge depth based on the casename in the filePath

    for filePath in filePaths:
        outfilename = filePath
        if "30" in outfilename:
            cp2_start = 8
        elif "60" in outfilename:
            cp2_start = 14
        elif "90" in outfilename:
            cp2_start = 20
        elif "120" in outfilename:
            cp2_start = 26 
            
        ###Loop through all variations and perform depth and environment calculations for the case that matches the filePath
        for w,wind in enumerate(wind_list):
            for c,CP1_temp in enumerate(temp_list):
                for k,CP2_temp in enumerate(temp_list):
                    for t,time in enumerate(time_list):
                        if f"CP1_{CP1_temp}_CP2_{CP2_temp}_{time:03}_{wind}" in filePath:

                            cp1_depth_list, cp2_depth_list,thv_bar1,thv_bar2, u_bar1,u_bar2,v_bar1,v_bar2 = calc_depth(filePath,wind,CP1_temp,CP2_temp,time,alt_list, thv, q)

                            test_array[w,c,k,t,0,:len(cp1_depth_list)] = np.array(cp1_depth_list)
                            test_array[w,c,k,t,1,:len(cp2_depth_list)] = np.array(cp2_depth_list)
                            
                            thv_array[w,c,k,t,0,:len(thv_bar1),:] = thv_bar1
                            thv_array[w,c,k,t,1,:len(thv_bar2),:] = thv_bar2
                            
                            u_array[w,c,k,t,0,:len(u_bar1),:] = u_bar1
                            u_array[w,c,k,t,1,:len(u_bar2),:] = u_bar2
                            
                            v_array[w,c,k,t,0,:len(u_bar1),:] = v_bar1
                            v_array[w,c,k,t,1,:len(u_bar2),:] = v_bar2
                            wind_list_pd.append(wind)
                            cp1_temp_list.append(CP1_temp)
                            cp2_temp_list.append(CP2_temp)
                            time_list_pd.append(time)
                            cp1_depth_list_all.append(cp1_depth_list)
                            cp2_depth_list_all.append(cp2_depth_list)

    columns = ['Wind', 'CP1_Temp', 'CP2_Temp', 'dt(CPs)', 'CP1_depth', 'CP2_depth']
    data = list(zip(np.array(wind_list_pd), np.array(cp1_temp_list), np.array(cp2_temp_list), np.array(time_list_pd),
            cp1_depth_list_all, cp2_depth_list_all))
    depths_dataframe = pd.DataFrame(data, columns = columns)
    
    return depths_dataframe, test_array, thv_array, u_array, v_array


def make_env_file(env_savePath, thv_array, u_array, v_array):
    '''teakes environmental virtual potential temperature, u, and v arrays and makes a file'''
    test_dataset = h5py.File(env_savePath, 'w')
    test_dataset.create_dataset('thv_bar', data=thv_array)
    test_dataset.create_dataset('u_bar', data=u_array)
    test_dataset.create_dataset('v_bar', data=v_array)
    test_dataset.close()
    
@TaskGenerator
def make_cpdepth_file(filePaths, savePath,alt_list,thv, q, mode =1):
    '''outlining function for making depths and environmental arrays and making files with them'''
    
    if mode ==1:
        depths_dataframe, depths_array,thv_array, u_array, v_array = make_cpdepth_array(filePaths,alt_list, thv, q)
    elif mode ==2:
         depths_dataframe, depths_array = make_cpdepth_array_avg(filePaths,alt_list, thv, q)
    elif mode ==3:
        depths_dataframe, depths_array,thv_array, u_array, v_array = make_cpdepth_array(filePaths,alt_list, thv, q)
        env_savePath = "./envir_cps_v2.h5"
        make_env_file(env_savePath, thv_array, u_array, v_array)
        return 
    
    test_dataset = h5py.File(savePath+".h5", 'w')
    test_dataset.create_dataset('depths', data=depths_array)
    test_dataset.close()
    
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

filePaths = sorted(glob.glob(f"/moonbow/cneumaie/CP_train_derecho/processed/*"))

thv_thresholds = np.arange(0.6,1.1,0.2)
quantile_test = np.arange(95,97,2)

for thv in thv_thresholds:
    for q in quantile_test:
        savePath = f"./cpdepths_test_files_time_v11_95/thv{round(thv,2):03}_q{q}"
        if (thv ==0.4) and (q==92):
            mode =1
        else:
            mode=1
        make_cpdepth_file(filePaths,savePath,alt, thv, q, mode = mode)