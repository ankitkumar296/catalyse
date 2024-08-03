import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import multiprocessing as mp
import psutil
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dense, Flatten

import numpy as np
import random
import pickle
import csv
from tensorflow.keras import layers, models

#  ===================================VGG16 Class ===============================================================


class vgg16_in():
    def __init__(self,prof_data):
        self.weight_set=False
        self.partition_done=False
        self.input_loaded=False
        self.layer_list=[]
        self.NO_OF_LAYERS= 22
        self.prof_data= prof_data
        
        # self.model = Sequential()
        
        # # Block 1
        # self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
        # self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # # Block 2
        # self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # # Block 3
        # self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # # Block 4
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # # Block 5
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # # Flatten
        # self.model.add(Flatten())
        
        # # Fully connected layers
        # self.model.add(Dense(4096, activation='relu'))
        # self.model.add(Dense(4096, activation='relu'))
        # self.model.add(Dense(1000, activation='softmax'))  # Output layer with 1000 classes for ImageNet
        # tf.keras.utils.plot_model(self.model, to_file='vgg16_model.png', show_shapes=True)
        # self.NO_OF_LAYERS= len(self.model.layers)
        # self.loadWeights()
        # self.make_partition()
        
    def model_info(self):
        self.no_of_layers=len(self.model.layers)
        print("\_____Number of Layers in Model: ",self.no_of_layers)
        print("\_____Weights are loaded: ",self.weight_set)
        print("\_____Input loaded to Model: ",self.input_loaded)
        print('\_____Partitioning Done: ',self.partition_done)
        
        print(type(self.model.layers[0]))
    def load_input(self):
        self.input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        self.input_loaded=True
    def loadWeights(self):
        self.model.load_weights('vgg16_imagenet_5epoch.h5')
        self.weight_set=True
        print("\_____Weights are loaded to the ")
        

    def save_layers_weight(self, dir_path):

        for i in range (self.NO_OF_LAYERS):
            weight=None
            weight=self.get_layer_weigth(i)
            # nw=np.array(weight)
            print(len(weight))
            if len(weight) <1:
                continue
            weights = weight[0].astype(np.float32)
            bias = weight[1].astype(np.float32)
            # token=[weights,bias]
            path_w=dir_path +'/layer_'+str(i)+'w.npy'
            path_b=dir_path +'/layer_'+str(i)+'b.npy'
            np.save(path_w,weights)
            np.save(path_b,bias)

    def get_layer_weigth(self, layer_id):

        layer_w=self.model.layers[layer_id].get_weights()
        # if len(layer_w) <1:
        #         pass
        # else:
        #     self.model.layers[layer_id].set_weights([layer_w[0],np.zeros(len(layer_w[1]))])

        return layer_w


    def load_layer_weights(self,dir_path):
        for i in range(9):
            if i in [1,3,5]:
                continue
            else:
                weight=np.load(f'{dir_path}/layer_{i}w.npy')
                bias=np.load(f'{dir_path}/layer_{i}b.npy')
                self.model.layers[i].set_weights([weight,bias])

    def load_layer(self, layer_id):
        self.dir_path='dnn_models/_dnn_saved_layers/vgg'
        fname=f'./{self.dir_path}/vgg16_layer_{layer_id}.pkl'
        with open(fname, 'rb') as f:
            layer_config = pickle.load(f)
            
        self.layer = tf.keras.layers.deserialize(config= layer_config['config'])
            
        # self.layer_list.append(layer)
        self.layer.set_weights(layer_config['weights'])
        return 0
        
    
    def load_layer__old(self, layer_id):

        
        
        match layer_id:
            case 0:
                self.layer=layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3))
                in_d=np.zeros(shape=(1,224, 224, 3)).astype(np.float32)
                self.layer(in_d)

            case 1:
                self.layer=Conv2D(64, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,224,224,64)).astype(np.float32)
                self.layer(in_d)

            case 2:
                self.layer=layers.MaxPooling2D((2, 2), strides=(2, 2))
                # in_d=np.zeros(shape=(1,15, 15, 32)).astype(np.float32)
                # self.layer(in_d)

            case 3:
                self.layer=layers.Conv2D(128, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,112, 112, 64)).astype(np.float32)
                self.layer(in_d)

            case 4:
                self.layer=layers.Conv2D(128, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,112, 112, 128)).astype(np.float32)
                self.layer(in_d)

            case 5:
                self.layer=layers.MaxPooling2D((2, 2), strides=(2, 2))
                # in_d=np.zeros(shape=(1,4, 4, 64)).astype(np.float32)
                # self.layer(in_d)

            case 6:
                self.layer=layers.Conv2D(256, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,56,56,128)).astype(np.float32)
                self.layer(in_d)

            case 7:
                self.layer=layers.Conv2D(256, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,56,56,256)).astype(np.float32)
                self.layer(in_d)

            case 8:
                self.layer=layers.Conv2D(256, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,56,56,256)).astype(np.float32)
                self.layer(in_d)
            case 9:
                self.layer=layers.MaxPooling2D((2, 2), strides=(2, 2))
                # in_d=np.zeros(shape=(1,64)).astype(np.float32)
                # self.layer(in_d)

            case 10:
                self.layer=layers.Conv2D(512, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,28,28,256)).astype(np.float32)
                self.layer(in_d)

            case 11:
                self.layer=layers.Conv2D(512, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,28,28,512)).astype(np.float32)
                self.layer(in_d)

            case 12:
                self.layer=layers.Conv2D(512, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,28,28,512)).astype(np.float32)
                self.layer(in_d)

            case 13:
                self.layer=layers.MaxPooling2D((2, 2), strides=(2, 2))
                # in_d=np.zeros(shape=(1,64)).astype(np.float32)
                # self.layer(in_d)
        
            case 14:
                self.layer=layers.Conv2D(512, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,14,14,512)).astype(np.float32)
                self.layer(in_d)

            case 15:
                self.layer=layers.Conv2D(512, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,14,14,512)).astype(np.float32)
                self.layer(in_d)

            case 16:
                self.layer=layers.Conv2D(512, (3, 3), activation='relu', padding='same')
                in_d=np.zeros(shape=(1,14,14,512)).astype(np.float32)
                self.layer(in_d)

            case 17:
                self.layer=layers.MaxPooling2D((2, 2), strides=(2, 2))
                # in_d=np.zeros(shape=(1,64)).astype(np.float32)
                # self.layer(in_d)

            case 18:
                self.layer=layers.Flatten()
                in_d=np.zeros(shape=(1,7,7,512)).astype(np.float32)
                self.layer(in_d)

            case 19:
                self.layer=layers.Dense(4096, activation='relu')
                in_d=np.zeros(shape=(1,25088)).astype(np.float32)
                self.layer(in_d)

            case 20:
                self.layer=layers.Dense(4096, activation='relu')
                in_d=np.zeros(shape=(1,4096)).astype(np.float32)
                self.layer(in_d)

            case 21:
                self.layer=layers.Dense(1000, activation='softmax')
                in_d=np.zeros(shape=(1,4096)).astype(np.float32)
                self.layer(in_d)


        if layer_id in [2,5,9,13,17,18]:
                 return 0

        weight=np.load(f'./vgg16_weights/layer_{layer_id}w.npy')
        bias=np.load(f'./vgg16_weights/layer_{layer_id}b.npy')
        self.layer.set_weights([weight,bias])






    def make_partition(self):
        self.NO_OF_LAYERS= len(self.model.layers)
        
        for i in range(self.NO_OF_LAYERS):
            self.temp_layer=self.model.layers[i]
            self.layer_list.append(self.temp_layer)
            
        self.partition_done = True
        print('\_______Partitioning Done')
        
    def execute_full_network(self):
        if not self.input_loaded:
            self.load_input()
        # print("I am in full Executioon")
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
        for lay in self.layer_list:
            st21=time.perf_counter()
            temp_out=lay(temp_out)
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
        temp_out=input_data
        layer_execution_time=0
        select_cpu_time=0
        layer_loading_time=0
        for lay in range(self.NO_OF_LAYERS):
            # print('In layer loop')
            t31=time.perf_counter()
            # best_proc_ele,cpu_log= self.get_best_cpu(lay,cpu_log,best_proc_ele)
            best_proc_ele,cpu_log= self.get_naive_best_cpu(lay,cpu_log)
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
            cpu_log[best_proc_ele]=True
        temp_out=temp_out.numpy()
        t2=time.perf_counter()
        elapsed_time=t2-t1
        return temp_out,elapsed_time,layer_execution_time,select_cpu_time,layer_loading_time,self.executed_map
               
    
    def execute_on_core(self,layer_id,input_data):
                
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out



       
# ====================================VGG16 End =================================================================
