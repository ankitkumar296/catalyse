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
        
        self.garbage_time=0
        # self.model = models.Sequential()
        # # # Normalization layer
        # self.model.add(layers.LayerNormalization(center=True , scale=True,input_shape=(66,200, 3)))

        # Convolutional layers
        # self.model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
        # self.model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
        # self.model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
        # self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # # Flatten layer
        # self.model.add(layers.Flatten())

        # # Fully connected layers
        # self.model.add(layers.Dense(100, activation='relu'))
        # self.model.add(layers.Dense(50, activation='relu'))
        # self.model.add(layers.Dense(10, activation='relu'))
        
        # # Output layer
        # self.model.add(layers.Dense(1))  # Output: steering angle
        # self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       
        # #install "pip install pydot-ng"  and "apt install graphviz" for ploting below model
        # tf.keras.utils.plot_model(self.model, to_file='pilotnet_model.png', show_shapes=True)
        
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
        
        self.dir_path='dnn_models/_dnn_saved_layers/pilotnet'
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
        return elt1,self.output
    
    def execute_full_partition(self):
        if not self.partition_done:
            self.make_partition()
        if not self.input_loaded:
            self.load_input()
            
        self.temp_res=self.input_data
        st2=time.perf_counter()
        for i in range(self.NO_OF_LAYERS):
            self.temp_res = self.layer_list[i](self.temp_res)
        ed2=time.perf_counter()
        self.temp_res=np.array(self.temp_res)
        elt2=ed2-st2
        elt2=float(elt2)
        
        return elt2,self.temp_res
    
    
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
            
            if any(cpu_log):
                if cand_layer == 0 :
                    my_list = self.prof_data[cand_layer][0]
                else:
                    my_list = self.prof_data[cand_layer][last_cpu]
                found_element = None
                for element in my_list:
                    if cpu_log[element] :
                        found_element = element
                        cpu_log[found_element]=False
                        break
                
                break
            else:
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
                
        # print(f"Returning CPU")
        return found_element, cpu_log

    def get_naive_best_cpu(self,cand_layer,cpu_log):
        
        while True:
            if any(cpu_log):
                if cand_layer == 0:
                    my_list = self.prof_data[0]
                else:

                    my_list = self.prof_data[cand_layer]
                
                found_element = None
                for element in my_list:
                    if cpu_log[element]:
                        found_element = element
                        cpu_log[found_element]=False
                        break
                break
            else:
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
                
        # print(f"Returning CPU")
        return found_element, cpu_log
    
    
    def execute_naive_cpu_layer_by_layer(self,input_data,cpu_log):
        self.executed_map=[]
        # print('Helo')
        t1=time.perf_counter()
        best_proc_ele=0
        p=psutil.Process()
        temp_out=input_data
        layer_execution_time=0
        select_cpu_time=0
        layer_loading_time=0
        for lay in range(self.NO_OF_LAYERS):
            # print('In layer loop')
            t31=time.perf_counter()
            # best_proc_ele,cpu_log= self.get_best_cpu(lay,cpu_log,best_proc_ele)
            best_proc_ele,cpu_log= self.get_naive_best_cpu(lay,cpu_log)
            # print(f"CPU Returned for layer {lay}")
            
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
            cpu_log[best_proc_ele]=True
            # try:
            #     cpu_log.append(best_proc_ele)
            # except Exception as e:
            #     # Handle the exception
            #     print("An error occurred:", e)
            # print(f'Layer {lay} Executed Syccessfully')
        temp_out=temp_out.numpy()
        t2=time.perf_counter()
        elapsed_time=t2-t1
        # p.cpu_affinity([])
        # print('Returning to CHILD')
        
        return temp_out,elapsed_time,layer_execution_time,select_cpu_time,layer_loading_time,self.executed_map
               
    
    def execute_on_core(self,layer_id,input_data):
                
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out
    
# ============================================PILOTNET End=======================================================================


# ================================================ Multiprocessing Worker Process Start ===========================================
