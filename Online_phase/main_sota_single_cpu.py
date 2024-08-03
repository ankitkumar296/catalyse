import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from tensorflow import keras
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Dropout,Lambda,Dense,Flatten,Input
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from tensorflow.keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras import Input as innp
import tensorflow.keras.backend as K
import tensorflow.keras as ker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dense, Flatten

import time
import psutil
import csv
import gc
import pickle

#   importing the Models

from dnn_models.sota_all.mtl_net_sota_all_cpu import *
from dnn_models.sota_all.pilotnet_sota_all_cpu import *
from dnn_models.sota_all.SD_sota_all_cpu import *
from dnn_models.sota_all.squeeznet_sota_all_cpu import *
from dnn_models.sota_all.vgg16_sota_all_cpu import *

def child(worker: int,cpu_log,mode:int,input_data) -> None:
    
    cp1=time.perf_counter()
    p = psutil.Process()
    
    p.cpu_affinity([worker])
    # print(f'Type of CPU_LOG {type(cpu_log)}', flush=True)
    # print(f'Current CPU_LOG : {cpu_log}',flush=True)
    
    print(f"Child #{worker}: Starting CPU intensive task now for 4 seconds on {p.cpu_affinity()}...", flush=True)
    
    obj=None
    cp2=time.perf_counter()
    if mode==11:
        obj=pilotnet()
    elif mode == 22:
        obj=vgg16_in()
    elif mode == 30:
        obj=custom_MTL()
    elif mode==18:
        obj=self_driving_mtl()
    elif mode == 44:
        obj=squeeznet()
        # r1=obj.make_partition()
    
    cp4=time.perf_counter()
    
    # r2=obj.execute_on_core(layer_id=layer_id, input_data=input_data, proc_ele=worker)
    # cpu_log.append(worker)
    # print(f'I am after appending worker')
    r2=obj.execute_full_network(input_data=input_data)
    # r2=obj.execute_layer_by_layer_sample_inp(inp=input_data)

    cp5=time.perf_counter()
    
    el1=cp2-cp1
    el2=cp4-cp2
    # el2=el2+r2[2]
    print(f"Child #{worker}: Finished CPU intensive task on {p.cpu_affinity()} for CNN-{mode} and takes affinity time : {el1} and creating object loading takes: {el2} ", flush=True)
    p.cpu_affinity([])
    #r2[0] = model output
    # r2[1] = elapsed time
    # r[2] = layer_ex time
    # r[3] =  CPU delay
    # r2[4] = layer_load_time
    # r[5] = executed CPU map
    # el2= contructor latency
    # cpu_log = current available CPUs
    return r2[0], r2[1],r2[2],r2[3],r2[4],r2[5],el2,cpu_log

def write2csv(file_name,write_token):
        with open(file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(write_token)


class update_class():
    def __init__(self,mode=0,read_file_name='read.csv',deadline=1) -> None:
        self.inter_result=None
        self.curent_layer=0
        self.lock=False
        self.mode=mode
        self.time_reads=[]
        self.exe_time=0
        self.obj_time=0
        self.prev_cpu=0
        self.cpu_track=[]
        self.read_file_name = read_file_name
        self.deadline=deadline
        self.deadline_setified=0
        
        # self.pool_obj=pool_obj
        
        
    #writing readings to csv file
    def write2csv(self,write_token):
        with open(self.read_file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(write_token)
    def updater(self,token):
        self.inter_result=token[0]
        self.exe_time=token[1]
        self.layer_ex_time =token[2]
        self.cpu_delay= token[3]
        self.layer_loading_time=token[4]
        self.cpu_track=token[5]
        self.obj_time = token[6]
        
        print(f"CNN-{self.mode} is completed succesfully with total execution time of : {self.exe_time} and object creation and loading time of {self.obj_time}", flush=True)
        print(f'CPU TRACK for Model : {self.cpu_track}')
        self.total_inf_time= self.exe_time + self.obj_time
        if self.total_inf_time <= self.deadline : self.deadline_setified=1
        self.write2csv([self.exe_time,self.cpu_delay,self.layer_loading_time,self.obj_time,self.total_inf_time,self.deadline_setified,self.cpu_track])
        print("CSV file UPDATED")


# obj1=custom_MTL()
# onj2=pilotnet()
# obj3= self_driving_mtl()
# obj4= squeeznet()
# obj5=vgg16_in()



def main(para=1) -> None:

    np.random.seed(18)

    NO_OF_CPU=24
    NO_OF_LAYERS_pilotnet=11
    NO_OF_LAYERS_vgg=22
    NO_OF_LAYERS_squeeznet=44
    NO_OF_LAYERS_self_driving_mtl=18
    NO_OF_LAYERS_custom_mtl=30
    
    
    vgg_deadline=2.2202
    squeeznet_deadline=0.2183
    pilotnet_deadline=0.1384
    custom_mtl_deadline=0.1831
    self_driving_mtl_deadline=0.176
        
    NO_OF_INFERENCE=para
    updators=[None]*NO_OF_INFERENCE
    inputs=[None]*NO_OF_INFERENCE
    
    main_et_file=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/main_ET.csv'
    # cpu_log=[i for i in range(24)]
    pool=mp.Pool()
    manager= mp.Manager()
    cpu_log= manager.list([True]*NO_OF_CPU)
    workers: int = pool._processes
    print(f"Running pool with {workers} workers",flush=True)

    # vgg_weights=manager.list(main_weight_list)
    # with open('./dnn_models/_profilings/greedy_profile/pilotnet_greedy_profile.pkl', 'rb') as file:
    #     pilotnet_profile = pickle.load(file)
    # with open('./dnn_models/_profilings/greedy_profile/vgg16_greedy_profile.pkl', 'rb') as file:
    #     vgg_profile = pickle.load(file)
    # with open('./dnn_models/_profilings/greedy_profile/squeeznet_greedy_profile.pkl', 'rb') as file:
    #     squeeznet_profile = pickle.load(file)
    # with open('./dnn_models/_profilings/greedy_profile/custom_mtl_greedy_profile.pkl', 'rb') as file:
    #     custom_mtl_profile = pickle.load(file)
    # with open('./dnn_models/_profilings/greedy_profile/self_driving_mtl_greedy_profile.pkl', 'rb') as file:
    #     self_driving_mtl_profile = pickle.load(file)
    
    
    def sleep_random_time(min_seconds, max_seconds):
        sleep_time = random.uniform(min_seconds, max_seconds)
        print(F'############## SLEEPING FOR {sleep_time} #############')
        time.sleep(sleep_time)
        
    min_sleep_time = 0.009
    max_sleep_time = 0.019
    
    mr1=time.perf_counter()
    print(f"Main programe starting time reading of perf_counter : {mr1}",flush=True)
    
    for inst in range(NO_OF_INFERENCE):
        sleep_random_time(min_sleep_time, max_sleep_time)
        curr_cpu=inst%NO_OF_CPU
        if inst in [0,4,8,16,20,18]:
            #VGG inference    
            updators[inst]=update_class(mode=22,read_file_name=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv',deadline=vgg_deadline)
            inputs[inst]=np.random.rand(1, 224, 224, 3).astype(np.float32)
            pool.apply_async(child, (curr_cpu,cpu_log,22,inputs[inst]),callback=updators[inst].updater)
        elif inst in [3,6,9,17,23,30]:
            #Squeeznet inference
            updators[inst]=update_class(mode=44,read_file_name=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv',deadline=squeeznet_deadline)
            inputs[inst]=np.random.rand(1, 224, 224, 3).astype(np.float32)
            pool.apply_async(child, (curr_cpu,cpu_log,44,inputs[inst]),callback=updators[inst].updater)
        elif inst in [2,10,12,14,18,22,24,26]:
            #Pilotnet inference
            updators[inst]=update_class(mode=11,read_file_name=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv',deadline=pilotnet_deadline)
            inputs[inst]=np.random.rand(1, 66, 200, 3).astype(np.float32)
            pool.apply_async(child, (curr_cpu,cpu_log,11,inputs[inst]),callback=updators[inst].updater)
        elif inst in [1,5,7,21,29,31]:
            #Custom MTL inference
            updators[inst]=update_class(mode=30,read_file_name=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv',deadline=custom_mtl_deadline)
            inputs[inst]=np.random.rand(1, 32, 32, 3).astype(np.float32)
            pool.apply_async(child, (curr_cpu,cpu_log,30,inputs[inst]),callback=updators[inst].updater)
        elif inst in [11,13,15,19,25,27]:
            #Self Driving MTL inference
            updators[inst]=update_class(mode=18,read_file_name=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv',deadline=self_driving_mtl_deadline)
            inputs[inst]=np.random.rand(1, 140, 208, 3).astype(np.float32)
            pool.apply_async(child, (curr_cpu,cpu_log,18,inputs[inst]),callback=updators[inst].updater)
    
    

    
    # Wait for children to finnish
    pool.close()
    pool.join()
    mr2=time.perf_counter()
    met=mr2-mr1
    write2csv(main_et_file,[met])
    print(f"Main programe end at {mr2} taking {met} seconds",flush=True)
    # pass


import gc
import sys
if __name__ == '__main__':
    para=int(sys.argv[1])
    for rr in range(1):
        gc.collect()
        time.sleep(1)
        main(para=para)
        gc.collect()