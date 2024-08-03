import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import multiprocessing as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

import numpy as np
import psutil
import time
import pickle
import csv
import random


# ============================================ PILOTNET Start ===================================================

class pilotnet():
    def __init__(self,prof_data=None,pilotnet_weights=None) -> None:
        self.weight_set=False
        self.partition_done=False
        self.input_loaded=False
        self.layer_list=[]
        self.NO_OF_LAYERS= 11
        self.layer=None
        self.prof_data= prof_data
        self.weight_list=pilotnet_weights
        
        
        self.model = models.Sequential()
        # # Normalization layer
        self.model.add(layers.LayerNormalization(center=True , scale=True,input_shape=(66,200, 3)))

        # Convolutional layers
        self.model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # Flatten layer
        self.model.add(layers.Flatten())

        # Fully connected layers
        self.model.add(layers.Dense(100, activation='relu'))
        self.model.add(layers.Dense(50, activation='relu'))
        self.model.add(layers.Dense(10, activation='relu'))
        
        # Output layer
        self.model.add(layers.Dense(1))  # Output: steering angle
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       
        #install "pip install pydot-ng"  and "apt install graphviz" for ploting below model
        # tf.keras.utils.plot_model(self.model, to_file='pilotnet_model.png', show_shapes=True)
        self.loadWeights()
        self.make_partition()
        # return self.model
        
    def loadWeights(self):
        self.model.load_weights('./pilotnet_model.h5')
        # model = load_model('model.h5')
        self.weight_set=True
        print("\_____Weights are loaded to the ")
        
    def model_info(self):
        self.no_of_layers=len(self.model.layers)
        print("\_____Number of Layers in Model: ",self.no_of_layers)
        print("\_____Weights are loaded: ",self.weight_set)
        print("\_____Input loaded to Model: ",self.input_loaded)
        print('\_____Partitioning Done: ',self.partition_done)
        
    def load_input(self):
        self.input_data = np.random.rand(1, 66, 200, 3).astype(np.float32)
        self.input_loaded=True
        
    def make_partition(self):
        self.NO_OF_LAYERS= len(self.model.layers)
        
        for i in range(self.NO_OF_LAYERS):
            self.temp_layer=self.model.layers[i]
            self.layer_list.append(self.temp_layer)
            
        self.partition_done = True
        print('\_______Partitioning Done')

    
    def save_pickeled_layers(self):
        if not self.weight_set:
            self.loadWeights()

        
        if not self.partition_done:
            self.make_partition()
        save_dir='pilotnet_pickle'
        for i in range(len(self.layer_list)):
            fname=f'./{save_dir}/pilotnet_layer_{i}.pkl'
            layer_weights_and_config = {
                'weights': self.layer_list[i].get_weights(),
                'config': tf.keras.layers.serialize(self.layer_list[i])}
            with open(fname, 'wb') as f:
                pickle.dump(layer_weights_and_config, f)
                
    def load_layer(self, layer_id):
        self.dir_path='pilotnet_pickle'
        fname=f'./{self.dir_path}/pilotnet_layer_{layer_id}.pkl'
        with open(fname, 'rb') as f:
            layer_config = pickle.load(f)
            
        self.layer = tf.keras.layers.deserialize(config= layer_config['config'])
            
        # self.layer_list.append(layer)
        self.layer.set_weights(layer_config['weights'])
        return self.layer
    
    def execute_full_network(self):
        if not self.input_loaded:
            self.load_input()
        print("I am in full Executioon")
        st1=time.perf_counter()
        self.output=self.model.predict(self.input_data)
        ed1=time.perf_counter()
        
        elt1=ed1-st1
        elt1=float(elt1)
        return elt1,self.output
        
    def execute_full_network_sample(self,imp):
        st1=time.perf_counter()        
        self.output=self.model.predict(imp)
        ed1=time.perf_counter()
        
        elt1=ed1-st1
        elt1=float(elt1)        
        return self.output,elt1,0,0,0,0
    
    def execute_full_partition(self,imp):
        if not self.partition_done:
            self.make_partition()
        # if not self.input_loaded:
        #     self.load_input()
            
        self.temp_res=imp
        st2=time.perf_counter()
        for i in range(self.NO_OF_LAYERS):
            self.temp_res = self.layer_list[i](self.temp_res)
        ed2=time.perf_counter()
        self.temp_res=np.array(self.temp_res)
        elt2=ed2-st2
        elt2=float(elt2)
        
        return self.temp_res,elt2,0,0,0,0
    
    
    def execute_layer_by_layer_sample_inp(self,inp):

        st1=time.perf_counter()
        # if not self.partition_done:
        #     self.make_partition()
        # if not self.input_loaded:
        #     self.load_input()
        self.executed_map=[]
        temp_out=inp
        layer_execution_time=0
        select_cpu_time=0
        for lay in range(11):
            st21=time.perf_counter()
            lla=self.load_layer(layer_id=lay)
            temp_out=lla(temp_out)
            ed21=time.perf_counter()
            el21=ed21-st21
            layer_execution_time += el21
               
        temp_out=temp_out.numpy()
        ed1=time.perf_counter()

        el1=ed1-st1
        return temp_out,el1,layer_execution_time,select_cpu_time,self.executed_map
    
    
    def get_best_cpu(self,cand_layer,cpu_log,last_cpu):
        # print(f"***********Current CPU_LOG {cpu_log}")
        
        while True:
            
            if len(cpu_log)<=0:
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
            else:
                if cand_layer == 0:
                    my_list = self.prof_data[0][0]
                else:

                    my_list = self.prof_data[cand_layer][last_cpu]
                
                found_element = None
                for element in my_list:
                    if element in cpu_log:
                        found_element = element
                        break
                cpu_log.remove(found_element)
                break
                
        # print(f"Returning CPU")
        return found_element, cpu_log

    
    
    
    def execute_layer_by_layer_sample(self,inp,cpu_log):
        self.executed_map=[]
        # print('Helo')
        t1=time.perf_counter()
        best_proc_ele=-1
        temp_out=inp
        layer_execution_time=0
        select_cpu_time=0
        layer_loading_time=0
        for lay in range(self.NO_OF_LAYERS):
            # print('In layer loop')
            t31=time.perf_counter()
            best_proc_ele,cpu_log= self.get_best_cpu(lay,cpu_log,best_proc_ele)
            # print(f"CPU Returned")
            p=psutil.Process()
            # print(f"################Changing the affinity from CPU {p.cpu_affinity()} to CPU NA for executing layer {lay}#####",flush=True)
            p.cpu_affinity([best_proc_ele])
            self.executed_map.append(best_proc_ele)
            t41=time.perf_counter()
            self.load_layer(lay)
            # print(f'Layer and Weights loaded for CNN-9 layer ID : {lay} succesfully',flush=True)
            t12=time.perf_counter()
            temp_out=self.layer(temp_out)
            t22=time.perf_counter()
            e2=t22-t12
            e3=t12-t31
            e4=t12-t41
            layer_execution_time +=e2
            select_cpu_time += e3
            layer_loading_time +=e4
            cpu_log.append(best_proc_ele)
        temp_out=temp_out.numpy()
        t2=time.perf_counter()
        elapsed_time=t2-t1
        
        return temp_out,elapsed_time,layer_execution_time,select_cpu_time,layer_loading_time,self.executed_map
               
    
    def execute_on_core(self,layer_id,input_data):
                
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out
    
# ============================================PILOTNET End=======================================================================


# ================================================ Multiprocessing Worker Process Start ===========================================

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
    elif mode == 7:
        obj=pilotnet()
        # r1=obj.make_partition()
    
    cp4=time.perf_counter()
    
    # r2=obj.execute_on_core(layer_id=layer_id, input_data=input_data, proc_ele=worker)
    # cpu_log.append(worker)
    # print(f'I am after appending worker')
    r2=obj.execute_full_network_sample(imp=input_data)
    # r2=obj.execute_full_partition(imp=input_data)
    # r2=obj.execute_layer_by_layer_sample(inp=input_data,cpu_log=cpu_log)
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
        self.write2csv([self.exe_time,self.cpu_delay,self.layer_loading_time,self.obj_time,self.total_inf_time,self.cpu_track])
        print("CSV file UPDATED")



def main(para=1) -> None:

    np.random.seed(18)

    NO_OF_CPU=24
    NO_OF_LAYERS=11
    
    NO_OF_INFERENCE=para
    updators=[None]*NO_OF_INFERENCE
    inputs=[None]*NO_OF_INFERENCE
    
    main_et_file=f'./readings/sota/predict_single/inst_set_{NO_OF_INFERENCE}_delay/main_ET.csv'
    # cpu_log=[i for i in range(24)]
    pool=mp.Pool()
    manager= mp.Manager()
    cpu_log= manager.list([i for i in range(NO_OF_CPU)] )
    workers: int = pool._processes
    print(f"Running pool with {workers} workers",flush=True)


    

    # vgg_weights=manager.list(main_weight_list)
    with open('pilotnet_vgg_greedy_profile_full.pkl', 'rb') as file:
        pilotnet_profile = pickle.load(file)
    
    
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
        updators[inst]=update_class(mode=11,read_file_name=f'./readings/sota/predict_single/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}.csv')
        inputs[inst]=np.random.rand(1, 66, 200, 3).astype(np.float32)
        sing_cp=inst%NO_OF_CPU
        pool.apply_async(child, (sing_cp,cpu_log,11,inputs[inst]),callback=updators[inst].updater)
    
    

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
    for rr in range(10):
        gc.collect()
        time.sleep(1)
        main(para=para)
        gc.collect()
    
# inp=np.random.rand(1, 66, 200, 3).astype(np.float32)
# ob=pilotnet()
# ob.loadWeights()
# o1=ob.execute_full_network_sample(imp=inp)
# o2=ob.execute_layer_by_layer_sample_inp(inp=inp)
# print(f'Out 1: {o1}')
# print(f'Out 2: {o2[0]}')
# print(f'Difference : {o2[0]-o1}')
    