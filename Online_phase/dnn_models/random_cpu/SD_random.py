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
import random

# =======================================SElf Driving MTL=========================

class self_driving_mtl():
    def __init__(self,self_driving_mtl_profile=None) -> None:
            
        
        self.weight_set=False
        self.partition_done=False
        self.NO_OF_LAYERS=18
        self.prof_data=self_driving_mtl_profile
        # input_shape = (140, 208, 3)
        # print('======================Creating input_layer=====================')
        # inputs = layers.Input(shape=input_shape ,name='0_input_layer')
        # # print('======================Creating BACKBONE=====================')
        # backbone_output = self.backbone_network(inputs)
        
        # classification_output = self.classification_branch(backbone_output)

        # detection_output = self.detection_branch(backbone_output)
        # # print('======================Creating MAIN MODEL=====================')
        # self.model = models.Model(inputs=inputs, outputs=[classification_output, detection_output])
        # # self.model=models.Model()

        # # Print model summary
        # # print(self.model.summary())
        # self.load_weights()
        # self.make_partition()

    
    def backbone_network(self,inputs):
        # print('--------------CREATING BACKBONE LAYER : 1 -------------------')
        x = layers.Conv2D(24, (12, 12), activation='relu', strides=2, padding='valid',name='1_conv2D_1')(inputs)
        # print('--------------CREATING BACKBONE LAYER : 2 -------------------')
        # try:
            # print('ahbflabhdlbdslhgsbflhsbl')
        x = layers.Conv2D(36, (8, 8), activation='relu', strides=2, padding='valid' ,name='2_conv2D_2')(x)
        # except Exception as e:
        #     print('-------------IN EXCEPTION CONTROL BLOCK')
        #     print(e)
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
    
    def get_best_cpu(self,cand_layer,cpu_log,last_cpu):
        # print(f"***********Current CPU_LOG {cpu_log}")
        
        while True:
            if any(cpu_log):
                true_indices = [index for index, value in enumerate(cpu_log) if value]

                # Choose a random index from the list of true indices
                found_element = None
                found_element = random.choice(true_indices)
                cpu_log[found_element]=False                
                break
            else:
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
                
        # print(f"Returning CPU")
        return found_element, cpu_log
    

    
    
    def execute_layer_by_layer_random(self,input_data,cpu_log):
        # print('--------In the Function--------------')
        t1=time.perf_counter()
        self.executed_map=[]
        best_proc_ele=0
        p=psutil.Process()
        out=buffer=input_data
        layer_execution_time=0
        select_cpu_time=0
        layer_loading_time=0
        for idx in range(self.NO_OF_LAYERS):
            # print('---------In LOOP---------------')
            curr_lay=None
            t31=time.perf_counter()
            
            best_proc_ele,cpu_log= self.get_best_cpu(idx,cpu_log,best_proc_ele)
            # print("Get the cPU")
            p.cpu_affinity([best_proc_ele])
            # print('Appended')
            self.executed_map.append(best_proc_ele)
            t41=time.perf_counter()
            
            curr_lay=self.load_layer(idx)
            # print('LAyers are Loaded')
            t12=time.perf_counter()
            if idx ==0:
                # out=curr_lay(out)
                out=out
            elif idx <= 7:
                # print('layer ging to execute')
                print(curr_lay)
                out=curr_lay(out)
                buffer=out
               
                # print(f'Executing index {idx} ')
            elif idx in [8,10,12,14,16]:
                out=curr_lay(out)
                # print(f'Executing index {idx}')

            elif idx in [9,11,13,15,17]:
                buffer=curr_lay(buffer)
            #     print(f'Executing index {idx} ')
            # print("After the if ELse condition")
            t22=time.perf_counter()
            curr_lay=None
            e2=t22-t12
            e3=t12-t31
            e4=t12-t41
            layer_execution_time +=e2
            select_cpu_time += e3
            layer_loading_time +=e4
            cpu_log[best_proc_ele]=True
        t2=time.perf_counter()
        elapsed_time=t2-t1
        # print(f'Elapsed Time: {el2}')
        return (out,buffer),elapsed_time,layer_execution_time,select_cpu_time,layer_loading_time,self.executed_map
    
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
    
    def load_layer(self, layer_id):
        self.dir_path='dnn_models/_dnn_saved_layers/self_driving_mtl'
        fname=f'./{self.dir_path}/self_driving_mtl_layer_{layer_id}.pkl'
        with open(fname, 'rb') as f:
            layer_config = pickle.load(f)
            
        self.layer = tf.keras.layers.deserialize(config= layer_config['config'])
            
        # self.layer_list.append(layer)
        self.layer.set_weights(layer_config['weights'])
        return self.layer
    
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
