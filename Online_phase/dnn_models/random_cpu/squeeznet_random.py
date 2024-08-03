import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import multiprocessing as mp

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras as ker
from tensorflow.keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from tensorflow.keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras import Input as innp
import tensorflow.keras.backend as K

import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import csv
import gc
import pickle
import random

# ============================================= Network Start ============================================

class squeeznet():
    def __init__(self,DNN_profile=None) -> None:
        
        self.map=[[-1,[0],[0]],[0,[0],[0]],[0,[0],[0]],
                  [1,[0],[0,1]],[1,[0],[0,2]],[0,[1],[0]],
                  [3,[0,2],[0,3]],[1,[0],[0,1]],[1,[0],[0,2]],
                  [0,[1],[0]],[2,[0,2],[0]],[2,[0,3],[0]],
                  [1,[0],[0,1]],[1,[0],[0,2]],[0,[1],[0]],
                  [2,[0,2],[0]],[1,[0],[0,3]],[1,[0],[0,1]],
                  [1,[0],[0,2]],[0,[1],[0]],[2,[0,2],[0]],
                  [2,[0,3],[0]],[1,[0],[0,1]],[1,[0],[0,2]],
                  [0,[1],[0]],[3,[0,2],[0,3]],[1,[0],[0,1]],
                  [1,[0],[0,2]],[0,[1],[0]],[2,[0,2],[0]],
                  [2,[0,3],[0]],[1,[0],[0,1]],[1,[0],[0,2]],
                  [0,[1],[0]],[2,[0,2],[0]],[1,[0],[0,3]],
                  [1,[0],[0,1]],[1,[0],[0,2]],[0,[1],[0]],
                  [2,[0,2],[0]],[2,[0,3],[0]],[0,[0],[0]],
                  [0,[0],[0]],[0,[0],[0]],[0,[0],[0]],[0,[0],[0]]]
        self.layer_list=[]
        self.weight_set=False
        self.partition_done = False
        # input_img = Input(shape=input_shape)
        # self.model=ker.Sequential
        self.prof_data=DNN_profile
        self.NO_OF_LAYERS=44
        
        # x = Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
        
        # x = self.create_fire_module(x, int(16*compression), name='fire2')
        # x = self.create_fire_module(x, int(16*compression), name='fire3', use_bypass=use_bypass)
        # x = self.create_fire_module(x, int(32*compression), name='fire4')
        
        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
        
        # x = self.create_fire_module(x, int(32*compression), name='fire5', use_bypass=use_bypass)
        # x = self.create_fire_module(x, int(48*compression), name='fire6')
        # x = self.create_fire_module(x, int(48*compression), name='fire7', use_bypass=use_bypass)
        # x = self.create_fire_module(x, int(64*compression), name='fire8')
        
        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool8')(x)
        
        # x = self.create_fire_module(x, int(64*compression), name='fire9', use_bypass=use_bypass)

        # if dropout_rate:
        #     x = Dropout(dropout_rate)(x)
            
        # x = self.output(x, nb_classes)
        
        # self.model=Model(inputs=(input_img), outputs=x)

        return None

    
    def output(self,x, nb_classes):
        x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
        x = GlobalAveragePooling2D(name='avgpool10')(x)
        x = Activation("softmax", name='softmax')(x)
        return x


    def create_fire_module(self,x, nb_squeeze_filter, name, use_bypass=False):
            
        nb_expand_filter = 4 * nb_squeeze_filter
        squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
        expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
        expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
        
        axis = self.get_axis()
        x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
        
        if use_bypass:
            x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
            
        return x_ret


    def get_axis(self):
        axis = -1 if K.image_data_format() == 'channels_last' else 1
        return axis
    
    def print_summary(self):
        print(self.model.summary())
    def load_weights(self):
        self.model.load_weights('./squeeznet_model.h5')
        self.weight_set=True
        return
    
    def load_input(self):
        self.input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        self.input_loaded=True
        return self.input_data
    def make_partition(self):
        self.layer_list=[]
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
        save_dir='./../pickled_layers'
        for i in range(len(self.layer_list)):
            fname=f'./{save_dir}/custum_mtl_layer_{i}.pkl'
            layer_weights_and_config = {
                'weights': self.layer_list[i].get_weights(),
                'config': tf.keras.layers.serialize(self.layer_list[i])}
            with open(fname, 'wb') as f:
                pickle.dump(layer_weights_and_config, f)
              
        return
    
    def load_layer(self, layer_id):
        self.dir_path='dnn_models/_dnn_saved_layers/squeeznet'
        fname=f'./{self.dir_path}/squeeznet_layer_{layer_id}.pkl'
        with open(fname, 'rb') as f:
            layer_config = pickle.load(f)
            
        self.layer = tf.keras.layers.deserialize(config= layer_config['config'])
            
        # self.layer_list.append(layer)
        self.layer.set_weights(layer_config['weights'])
        return self.layer
      
    def execute_full(self, input_data):
        st1=time.perf_counter()
        out=self.model.predict(input_data)
        et1=time.perf_counter()
        el=et1-st1
        # print(el)
        return out,el,0,0,0,0
    def print_layrs(self):
        i=0
        for lay in self.model.layers:
            self.layer_list.append(lay)
            print(f'Index: {i} --> {lay.name}')
            i+=1
            
            
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

    def get_naive_best_cpu(self,cand_layer,cpu_log):
        while True:
            if len(cpu_log)<=0:
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
            else:
                if cand_layer == 0:
                    my_list = self.prof_data[0]
                else:
                    my_list = self.prof_data[cand_layer]
                found_element = None
                for element in my_list:
                    if cpu_log[element]:
                        found_element = element
                        cpu_log[found_element]=False
                        # try:
                        #     cpu_log.remove(found_element)
                        # except Exception as e:
                        #     # Handle the exception
                        #     print("An error occurred:", e)
                        break
                # cpu_log.remove(found_element)
                
                
                break
                
        # print(f"Returning CPU")
        return found_element, cpu_log
    
    def execute_layer_by_layer_sample_inp(self, input_data,cpu_log):
        st2=time.perf_counter()
        
        self.buffer=[None,None,None,None]
        self.buffer[0]=input_data
        
        
        for idx in range(self.NO_OF_LAYERS):
            
            curr_lay=self.model.layers[idx]
            # print(f"Executing Layer -> {idx}")
            match self.map[idx][0]:
                
                case -1:
                    self.buffer[self.map[idx][2][0]]=input_data
                case 0:
                    self.buffer[self.map[idx][2][0]]=curr_lay(self.buffer[self.map[idx][1][0]])
                case 1:
                    self.buffer[self.map[idx][2][0]]=self.buffer[self.map[idx][2][1]]=curr_lay(self.buffer[self.map[idx][1][0]])
                case 2:
                    self.buffer[self.map[idx][2][0]]=curr_lay([self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]])
                case 3:
                    self.buffer[self.map[idx][2][0]]=self.buffer[self.map[idx][2][1]]=curr_lay([self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]])
                    
        et2=time.perf_counter()
        el2=et2-st2
        # print(el2)
        return self.buffer[0].numpy(),el2,0,0,0,0
    
    def execute_layer_by_layer_random(self, input_data,cpu_log):
        
        # print("--------------First_-----------")
        t1=time.perf_counter()
        self.executed_map=[]
        best_proc_ele=0
        p=psutil.Process()
        self.buffer=[None,None,None,None]
        self.buffer[0]=input_data
        layer_execution_time=0
        select_cpu_time=0
        layer_loading_time=0
        
        for idx in range(self.NO_OF_LAYERS):
            # print("---------------Entering LOOP-----------------")
            t31=time.perf_counter()
            
            best_proc_ele,cpu_log= self.get_best_cpu(idx,cpu_log,best_proc_ele)
            p.cpu_affinity([best_proc_ele])
            self.executed_map.append(best_proc_ele)
            t41=time.perf_counter()
            curr_lay=self.load_layer(idx)
            t12=time.perf_counter()
            match self.map[idx][0]:
                
                case -1:
                    self.buffer[self.map[idx][2][0]]=input_data
                case 0:
                    self.buffer[self.map[idx][2][0]]=curr_lay(self.buffer[self.map[idx][1][0]])
                case 1:
                    self.buffer[self.map[idx][2][0]]=self.buffer[self.map[idx][2][1]]=curr_lay(self.buffer[self.map[idx][1][0]])
                case 2:
                    self.buffer[self.map[idx][2][0]]=curr_lay([self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]])
                case 3:
                    self.buffer[self.map[idx][2][0]]=self.buffer[self.map[idx][2][1]]=curr_lay([self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]])
            t22=time.perf_counter()
            
            e2=t22-t12
            e3=t12-t31
            e4=t12-t41
            layer_execution_time +=e2
            select_cpu_time += e3
            layer_loading_time +=e4
            cpu_log[best_proc_ele]=True
        out=self.buffer[0].numpy()
        t2=time.perf_counter()
        elapsed_time=t2-t1
        
        return out,elapsed_time,layer_execution_time,select_cpu_time,layer_loading_time,self.executed_map
    
    def get_input_list(self, input_data):
        st2=time.perf_counter()
        self.buffer=[None,None,None,None]
        self.buffer[0]=input_data
        self.input_list=[0]*44
        print(f'Number_of_layers : {len(self.model.layers)}')
        for idx in range(len(self.model.layers)):
            
            curr_lay=self.model.layers[idx]
            
            match self.map[idx][0]:
                
                case -1:
                    self.input_list[idx]=input_data
                    self.buffer[self.map[idx][2][0]]=input_data
                case 0:
                    self.input_list[idx]=self.buffer[self.map[idx][1][0]]
                    self.buffer[self.map[idx][2][0]]=curr_lay(self.buffer[self.map[idx][1][0]])
                case 1:
                    self.input_list[idx]=self.buffer[self.map[idx][1][0]]
                    self.buffer[self.map[idx][2][0]]=self.buffer[self.map[idx][2][1]]=curr_lay(self.buffer[self.map[idx][1][0]])
                case 2:
                    self.input_list[idx]=[self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]]
                    self.buffer[self.map[idx][2][0]]=curr_lay([self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]])
                case 3:
                    self.input_list[idx]=[self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]]
                    self.buffer[self.map[idx][2][0]]=self.buffer[self.map[idx][2][1]]=curr_lay([self.buffer[self.map[idx][1][0]],self.buffer[self.map[idx][1][1]]])
                    
        print(f"Input List Lenght : {len(self.input_list)}")
        et2=time.perf_counter()
        el2=et2-st2
        print(el2)
        return self.input_list
    
    def execute_on_core(self,layer_id,input_data,dummy_data):
        dummy_data=dummy_data
        # print(self.layer_list[layer_id].name)
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out

# ++++++++++++++++++++++++++++++++++++++++++++ Network End +++++++++++++++++++++++++++++++++

# ===================================== Multi processing Child ==============================
