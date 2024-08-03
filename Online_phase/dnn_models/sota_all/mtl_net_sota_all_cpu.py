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

import time
import psutil
import csv
import gc
import pickle


# =========================================== Custom MTL DNN Start ===========================================

class custom_MTL():
    def __init__(self) -> None:
        # print("Entering Constructor")
        self.layer_list=[]
        self.weight_set=False
        self.partition_done = False
        self.NO_OF_LAYERS=30
        
        self.angle_classifier_no=4
        self.cifar10_classifier_no=10
        self.inputs = keras.Input((32, 32, 3))
        self.conv_base = self.get_convbase(self.inputs)
        # print('----------Mid Contructor------------------')
        try:
            self.angle_classifier = self.get_classifier(self.conv_base, self.angle_classifier_no, "angle",count=15)
        except Exception as e:
            # Handle the exception
            print("An error occurred:", e)
        
        # print('----------Mid Contructor------------------')
        self.cifar10_classifier = self.get_classifier(self.conv_base, self.cifar10_classifier_no, "cifar10",count=22)
        self.model = Model(
            inputs=self.inputs, 
            outputs=[self.cifar10_classifier, self.angle_classifier]
        )
        # print('----------Mid Contructor------------------')
        self.load_weights()
        self.make_partition()
        # print("COnstructor Exit")
        # tf.keras.utils.plot_model(self.model, to_file='final_mtl_model.png', show_shapes=True,show_layer_names=True)
    
   
    def get_convbase(self,inputs):
    
    # reg = keras.regularizers.l2(1e-4)
    
    # initializer = keras.initializers.HeNormal()


        x = Conv2D(16, (3, 3), padding="same",name='1_conv2D_1')(inputs)
        x = Activation("relu",name='2_activation_1')(x)
        x = BatchNormalization(axis=-1,name='3_batch_norm_1')(x)
        x = MaxPooling2D(pool_size=(3, 3),name='4_maxPool_1')(x)
        x = Dropout(0.25,name='5_dropout_1')(x)
        
        x = Conv2D(32, (3, 3), padding="same",name='6_conv2D_2')(x)
        x = Activation("relu",name='7_activation_2')(x)
        x = BatchNormalization(axis=-1,name='8_batch_norm_2')(x)
        x = MaxPooling2D(pool_size=(2, 2),name='9_maxPool_2')(x)
        x = Dropout(0.25,name='10_dropout_2')(x)
        
        x = Conv2D(32, (3, 3), padding="same",name='11_conv2D_3')(x)
        x = Activation("relu",name='12_activation_3')(x)
        x = BatchNormalization(axis=-1,name='13_batch_norm_3')(x)
        x = MaxPooling2D(pool_size=(2, 2),name='14_maxPool_3')(x)
        x = Dropout(0.25,name='15_dropout_3')(x)
        
        return x
    
    def get_classifier(self,x, class_no, name,count):
    
        x = Flatten(name=f'{count+1}_layer')(x)
        x = Dense(128,name=f'{count+2}_layer')(x)
        x = Activation("relu",name=f'{count+3}_layer')(x)
        x = BatchNormalization(name=f'{count+4}_layer')(x)
        x = Dropout(0.5,name=f'{count+5}_layer')(x)
        x = Dense(class_no,name=f'{count+6}_layer')(x)
        x = Activation("softmax", name=name)(x)

        return x

    def load_weights(self):
        self.model.load_weights('./dnn_models/_model_files/custom_mtl_model.h5')
        self.weight_set=True
        return
    def load_input(self):
        self.input_data = np.random.rand(1, 32, 32, 3).astype(np.float32)
        self.input_loaded=True
        return self.input_data
        
    def save_pickeled_layers(self):
        if not self.weight_set:
            self.loadWeights()

        
        if not self.partition_done:
            self.make_partition()
        save_dir='./../pickle_layers'
        for i in range(len(self.layer_list)):
            fname=f'./{save_dir}/custum_mtl_layer_{i}.pkl'
            layer_weights_and_config = {
                'weights': self.layer_list[i].get_weights(),
                'config': tf.keras.layers.serialize(self.layer_list[i])}
            with open(fname, 'wb') as f:
                pickle.dump(layer_weights_and_config, f)
             
    def load_layer(self, layer_id):
        self.dir_path='pickle_layers'
        fname=f'./{self.dir_path}/custum_mtl_layer_{layer_id}.pkl'
        with open(fname, 'rb') as f:
            layer_config = pickle.load(f)
            
        self.layer = tf.keras.layers.deserialize(config= layer_config['config'])
            
        # self.layer_list.append(layer)
        self.layer.set_weights(layer_config['weights'])
        return self.layer   
                
    def make_partition(self):
        self.layer_list=[]
        self.NO_OF_LAYERS= len(self.model.layers)
        
        for i in range(self.NO_OF_LAYERS):
            self.temp_layer=self.model.layers[i]
            self.layer_list.append(self.temp_layer)
            
        self.partition_done = True
        print('\_______Partitioning Done')
        
    def execute_full_network(self,input_data):
        print("--------------------entring the executing function----------------")
        st1=time.perf_counter()
        out=self.model.predict(input_data)
        et1=time.perf_counter()
        el1=et1-st1
        print(f'Elapsed Time: {el1}')
        return out,el1,0,0,0,0
    
    def print_lays(self):
        for lay in self.model.layers:
            self.layer_list.append(lay)
            print(lay)
        print(f'Total number of Layers : {len(self.layer_list)}')
        
        
    def execute_layer_by_layer_sample_inp(self,input_data):
        st2=time.perf_counter()
        out=buffer=input_data
        for idx in range(1,len(self.model.layers)):
            
            if idx <= 15:
                # print(f'Executing index {idx} : {self.model.layers[idx]}')
                out=buffer=self.layer_list[idx](out)
            elif idx in [16,18,20,22,24,26,28]:
                # print(f'Executing index {idx} : {self.model.layers[idx]}')
                out=self.layer_list[idx](out)
            elif idx in [17,19,21,23,25,27,29]:
                # print(f'Executing index {idx} : {self.model.layers[idx]}')
                buffer=self.layer_list[idx](buffer)
        et2=time.perf_counter()
        el2=et2-st2
        # print(f'Elapsed Time: {el2}')
        out=(out,buffer)
        return out,el2,0,0,0,0
    
    def execute_lbl_serial(self,input_data):
        st2=time.perf_counter()
        out=buffer=input_data
        for idx in range(1,self.NO_OF_LAYERS):
            
            curr_lay=self.load_layer(idx)
            if idx <= 15:
                # print(f'Executing index {idx} : {self.model.layers[idx]}')
                out=buffer=curr_lay(out)
            elif idx in [16,18,20,22,24,26,28]:
                # print(f'Executing index {idx} : {self.model.layers[idx]}')
                out=curr_lay(out)
            elif idx in [17,19,21,23,25,27,29]:
                # print(f'Executing index {idx} : {self.model.layers[idx]}')
                buffer=curr_lay(buffer)
        et2=time.perf_counter()
        el2=et2-st2
        print(f'Elapsed Time: {el2}')
        return out,buffer
    
    def get_input_list(self,input_data):
        self.input_list=[]
        st2=time.perf_counter()
        out=buffer=input_data
        self.input_list.append(input_data)
        for idx in range(1,len(self.model.layers)):
            # print(f'Executing index {idx} :  {self.model.layers[idx]}')
            if idx <= 15:
                self.input_list.append(out)
                out=buffer=self.model.layers[idx](out)
            elif idx in [16,18,20,22,24,26,28]:
                self.input_list.append(out)
                out=self.model.layers[idx](out)
            elif idx in [17,19,21,23,25,27,29]:
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
        


# =========================================== Custom MTL DNN End =============================================
# ================================================ Multiprocessing Worker Process Start ===========================================
