
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import psutil
import csv
import gc

class self_driving_mtl():
    def __init__(self) -> None:
            
        
        self.weight_set=False
        self.partition_done=False
        
        input_shape = (140, 208, 3)

        # Create input layer
        inputs = layers.Input(shape=input_shape ,name='0_input_layer')

        # Backbone network
        backbone_output = self.backbone_network(inputs)
        
        classification_output = self.classification_branch(backbone_output)

        # Detection branch
        detection_output = self.detection_branch(backbone_output)

        # Create model
        self.model = models.Model(inputs=inputs, outputs=[classification_output, detection_output])

        # Print model summary
        # print(self.model.summary())
        self.load_weights()
        self.make_partition()

    
    def backbone_network(self,inputs):
        x = layers.Conv2D(24, (12, 12), activation='relu', strides=2, padding='valid',name='1_conv2D_1')(inputs)
        x = layers.Conv2D(36, (8, 8), activation='relu', strides=2, padding='valid' ,name='2_conv2D_2')(x)
        x = layers.Conv2D(48, (8, 8), activation='relu', strides=2, padding='valid' ,name='3_conv2D_2')(x)
        x = layers.Dropout(0.5 ,name='4_dropout')(x)
        x = layers.Conv2D(64, (5, 5), activation='relu', padding='valid', dilation_rate=2 ,name='5_conv2D_4')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='valid' ,name='6_conv2D_5')(x)
        x = layers.Flatten(name='7_flatten')(x)
        return x
    
    def classification_branch(self,backbone_output):
        x = layers.Dense(100, activation='relu',name='8_dense_12')(backbone_output)
        x = layers.Dense(50, activation='relu' ,name='10_dense_13')(x)
        x = layers.Dense(10, activation='relu' ,name='12_dense_14')(x)
        x = layers.Dense(1, activation='relu' ,name='14_dense_15')(x)
        outputs = layers.Activation( activation='relu',  name='yaw_rate')(x)
        return outputs
    
    def detection_branch(self,backbone_output):
        x = layers.Dense(100, activation='relu' ,name='9_dense_22')(backbone_output)
        x = layers.Dense(50, activation='relu' ,name='11_dense_23')(x)
        x = layers.Dense(10, activation='relu' ,name='13_dense_24')(x)
        x = layers.Dense(1, activation='relu' ,name='15_dense_25')(x)
        outputs = layers.Activation( activation='relu', name='speed')(x)
        return outputs

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
        print(f"Elapsed Time : {el1}")
        
        return out
    
    def execute_lbl(self,input_data):
        st2=time.perf_counter()
        out=buffer=input_data
        for idx in range(1,len(self.model.layers)):
            # print(f'Executing: {self.model.layers[idx]}')
            if idx <= 7:
                out=buffer=self.model.layers[idx](out)
            elif idx in [8,10,12,14,16]:
                out=self.model.layers[idx](out)
            elif idx in [9,11,13,15,17]:
                buffer=self.model.layers[idx](buffer)
        et2=time.perf_counter()
        el2=et2-st2
        print(f'Elapsed Time: {el2}')
        return out,buffer
    
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
    
    def execute_on_core(self,layer_id,input_data,dummy_data):
        dummy_data=dummy_data
        # print(self.layer_list[layer_id].name)
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out

NO_OF_LAYERS=18
NO_OF_CPU=24


image= np.random.rand(1, 140,208,3)

t1=time.perf_counter()
obj=self_driving_mtl()

# o1= obj.full_execution(input_data=image)
o2= obj.execute_lbl(input_data=image)

t2=time.perf_counter()
el1=t2-t1
print(f'Elapsed Time  : {el1}  Seconds')




