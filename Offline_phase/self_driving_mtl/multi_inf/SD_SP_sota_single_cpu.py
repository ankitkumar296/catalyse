import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import multiprocessing as mp

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import psutil
import csv
import gc
import sys
import random
# =======================================SElf Driving MTL=========================

class self_driving_mtl():
    def __init__(self) -> None:
            
        
        self.weight_set=False
        self.partition_done=False
        
        input_shape = (140, 208, 3)
        # print('======================Creating input_layer=====================')
        inputs = layers.Input(shape=input_shape ,name='0_input_layer')
        # print('======================Creating BACKBONE=====================')
        backbone_output = self.backbone_network(inputs)
        
        classification_output = self.classification_branch(backbone_output)

        detection_output = self.detection_branch(backbone_output)
        # print('======================Creating MAIN MODEL=====================')
        self.model = models.Model(inputs=inputs, outputs=[classification_output, detection_output])
        # self.model=models.Model()

        # Print model summary
        # print(self.model.summary())
        self.load_weights()
        self.make_partition()

    
    def backbone_network(self,inputs):
        # print('--------------CREATING BACKBONE LAYER : 1 -------------------')
        x = layers.Conv2D(24, (12, 12), activation='relu', strides=2, padding='valid',name='1_conv2D_1')(inputs)
        # print('--------------CREATING BACKBONE LAYER : 2 -------------------')
        try:
            # print('ahbflabhdlbdslhgsbflhsbl')
            x = layers.Conv2D(36, (8, 8), activation='relu', strides=2, padding='valid' ,name='2_conv2D_2')(x)
        except Exception as e:
            print('-------------IN EXCEPTION CONTROL BLOCK')
            print(e)
        # print('--------------CREATING BACKBONE LAYER : 3 -------------------')
        x = layers.Conv2D(48, (8, 8), activation='relu', strides=2, padding='valid' ,name='3_conv2D_2')(x)
        # print('--------------CREATING BACKBONE LAYER : 4 -------------------')
        x = layers.Dropout(0.5 ,name='4_dropout')(x)
        # print('--------------CREATING BACKBONE LAYER : 5 -------------------')
        x = layers.Conv2D(64, (5, 5), activation='relu', padding='valid', dilation_rate=2 ,name='5_conv2D_4')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='valid' ,name='6_conv2D_5')(x)
        # print('--------------CREATING BACKBONE LAYER : 7 -------------------')
        x = layers.Flatten(name='7_flatten')(x)
        return x
    
    def classification_branch(self,backbone_output):
        x = layers.Dense(100, activation='relu',name='8_dense_12')(backbone_output)
        x = layers.Dense(50, activation='relu' ,name='10_dense_13')(x)
        x = layers.Dense(10, activation='relu' ,name='12_dense_14')(x)
        x = layers.Dense(1, activation='relu' ,name='14_dense_15')(x)
        outputs1 = layers.Activation( activation='relu',  name='yaw_rate')(x)
        return outputs1
    
    def detection_branch(self,backbone_output):
        x = layers.Dense(100, activation='relu' ,name='9_dense_22')(backbone_output)
        x = layers.Dense(50, activation='relu' ,name='11_dense_23')(x)
        x = layers.Dense(10, activation='relu' ,name='13_dense_24')(x)
        x = layers.Dense(1, activation='relu' ,name='15_dense_25')(x)
        outputs2 = layers.Activation( activation='relu', name='speed')(x)
        return outputs2

    def load_weights(self):
        self.model.load_weights('./self_driving_mtl_model.h5')
        self.weight_set=True
    
    def print_layers(self):
        # self.layer_list=[]
        
        for lay in self.model.layers:
            print(lay.name)
            # self.layer_list.append(lay)
    
    def full_execution(self,input_data):
        if not self.weight_set:
            self.load_weights()
        
        st1=time.perf_counter()
        out=self.model.predict(input_data)
        et1=time.perf_counter()
        el1=et1-st1
        # print(f"Elapsed Time : {el1}")
        
        return out,el1,0,0,0,0
    
    def execute_layer_by_layer_sample_inp(self,input_data):
        st2=time.perf_counter()
        out=buffer=input_data
        for idx in range(1,self.NO_OF_LAYERS):
            # print(f'Executing: {self.layer_list[idx]}')
            if idx <= 7:
                out=buffer=self.layer_list[idx](out)
            elif idx in [8,10,12,14,16]:
                out=self.layer_list[idx](out)
            elif idx in [9,11,13,15,17]:
                buffer=self.layer_list[idx](buffer)
        et2=time.perf_counter()
        el2=et2-st2
        # print(f'Elapsed Time: {el2}')
        out=(out,buffer)
        return out,el2,0,0,0,0
    
    def make_partition(self):
        self.layer_list=[]
        self.NO_OF_LAYERS= len(self.model.layers)
        
        for i in range(self.NO_OF_LAYERS):
            self.temp_layer=self.model.layers[i]
            self.layer_list.append(self.temp_layer)
            
        self.partition_done = True
    
    def save_pickeled_layers(self):
        # if not self.weight_set:
        self.load_weights()

        
        # if not self.partition_done:
        self.make_partition()
        save_dir='../pickle_layers'
        for i in range(len(self.layer_list)):
            fname=f'./{save_dir}/self_driving_mtl_layer_{i}.pkl'
            layer_weights_and_config = {
                'weights': self.layer_list[i].get_weights(),
                'config': tf.keras.layers.serialize(self.layer_list[i])}
            with open(fname, 'wb') as f:
                pickle.dump(layer_weights_and_config, f)
                
    def get_input_list(self,input_data):
        self.input_list=[]
        st2=time.perf_counter()
        out=buffer=input_data
        self.input_list.append(input_data)
        for idx in range(1,len(self.model.layers)):
            # print(f'Executing index {idx} :  {self.model.layers[idx]}')
            if idx <= 7:
                self.input_list.append(out)
                out=buffer=self.model.layers[idx](out)
            elif idx in [8,10,12,14,16]:
                self.input_list.append(out)
                out=self.model.layers[idx](out)
            elif idx in [9,11,13,15,17]:
                self.input_list.append(buffer)
                buffer=self.model.layers[idx](buffer)
        et2=time.perf_counter()
        el2=et2-st2
        print(f'Elapsed Time: {el2}')
        return self.input_list
    
    def execute_on_core(self,layer_id,input_data):
        # dummy_data=dummy_data
        # print(self.layer_list[layer_id].name)
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out

# ================================Self_Driving MTL========================================


# ================================================ Multiprocessing Worker Process Start ===========================================

def child(worker: int,cpu_log,mode:int,input_data) -> None:
    # gc.enable()
    cp1=time.perf_counter()
    p = psutil.Process()
    # print('Enter the child')
    p.cpu_affinity([worker])
    # print(f'Type of CPU_LOG {type(cpu_log)}', flush=True)
    # print(f'Current CPU_LOG : {cpu_log}',flush=True)
    
    print(f"Child #{worker}: Starting CPU intensive task now for 4 seconds on {p.cpu_affinity()}...", flush=True)
    
    # obj=None
    cp2=time.perf_counter()
    if mode==18:
        obj=self_driving_mtl()
    elif mode == 7:
        obj=self_driving_mtl()
        # r1=obj.make_partition()
    # print('object_created')
    cp4=time.perf_counter()
    
    # r2=obj.execute_on_core(layer_id=layer_id, input_data=input_data, proc_ele=worker)
    # cpu_log.append(worker)
    # print(f'I am after appending worker')
    # r2=obj.execute_full_network_sample(imp=input_data)
    # r2=obj.full_execution(input_data=input_data)
    
    r2=obj.execute_layer_by_layer_sample_inp(input_data=input_data)

    cp5=time.perf_counter()
    
    el1=cp2-cp1
    el2=cp4-cp2
    el3=cp5-cp2
    # el2=el2+r2[2]
    print(f"Child #{worker}: Finished CPU intensive task on {p.cpu_affinity()} for CNN-{mode} and takes affinity time : {el1} and creating object loading takes: {el2} TOTAL: {el3} ", flush=True)
    p.cpu_affinity([])
    #r2[0] = model output
    # r2[1] = elapsed time
    # r[2] = layer_ex time
    # r[3] =  CPU delay
    # r2[4] = layer_load_time
    # r[5] = executed CPU map
    # el2= contructor latency
    # cpu_log = current available CPUs
    # gc.collect()
    
    return r2[0], r2[1],r2[2],r2[3],r2[4],r2[5],el2


# ===================================================Multiprocessing Worker Process End ===========================================


def write2csv(file_name,write_token):
        with open(file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(write_token)

class update_class():
    def __init__(self,mode=0,read_file_name='read.csv') -> None:
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
        # self.pool_obj=pool_obj
        gc.collect()
        
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
        self.total_inf_time= self.exe_time + self.obj_time
        print(f"CNN-{self.mode} is completed succesfully with total execution time of : {self.total_inf_time} and object creation and loading time of {self.obj_time}", flush=True)
        print(f'CPU TRACK for Model : {self.cpu_track}')
        
        self.write2csv([self.exe_time,self.cpu_delay,self.layer_loading_time,self.obj_time,self.total_inf_time,self.cpu_track])
        print("CSV file UPDATED")






def main(para=1) -> None:

    np.random.seed(18)
    # tf.Session()
    NO_OF_CPU=24
    NO_OF_LAYERS=18
    NO_OF_INFERENCE=para
    updators=[None]*NO_OF_INFERENCE
    inputs=[None]*NO_OF_INFERENCE
    
    main_et_file=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/main_ET_random.csv'
    # cpu_log=[i for i in range(24)]
    pool=mp.Pool()
    manager= mp.Manager()
    cpu_log= manager.list([i for i in range(NO_OF_CPU)] )
    workers: int = pool._processes
    print(f"Running pool with {workers} workers",flush=True)


    
    # vgg_weights=manager.list(main_weight_list)
    # with open('pilotnet_vgg_greedy_profile_full.pkl', 'rb') as file:
    #     custom_mtl_profile = pickle.load(file)
    
    def sleep_random_time(min_seconds, max_seconds):
        sleep_time = random.uniform(min_seconds, max_seconds)
        print(F'############## SLEEPING FOR {sleep_time} #############')
        time.sleep(sleep_time)
        
    min_sleep_time = 0.19
    max_sleep_time = 0.39
    mr1=time.perf_counter()
    print(f"Main programe starting time reading of perf_counter : {mr1}",flush=True)
    
    for inst in range(NO_OF_INFERENCE):
        # sleep_random_time(min_sleep_time, max_sleep_time)
        updators[inst]=update_class(mode=18,read_file_name=f'./readings/sota_single_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv')
        inputs[inst]=np.random.rand(1, 140, 208, 3).astype(np.float32)
        sing_cp=inst%NO_OF_CPU
        pool.apply_async(child, (sing_cp,cpu_log,18,inputs[inst]),callback=updators[inst].updater)

    
    
    
    # gc.collect()
    # Wait for children to finnish
    pool.close()
    pool.join()
    mr2=time.perf_counter()
    # tf.Session.close()
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



