# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from tensorflow import keras
# from tensorflowkeras.utils.vis_utils import plot_model
import random
from sklearn.model_selection import train_test_split
from scipy import ndimage
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Dropout,Lambda,Dense,Flatten,Input

import tensorflow as tf

import time
import psutil
import csv
import gc

# %%
class custom_MTL():
    def __init__(self) -> None:
        
        self.layer_list=[]
        x_shape, y_angle_shape, y_cifar10_shape=(50000, 32, 32, 3), (50000, 4), (50000, 10)
        angle_classifier_no=4
        cifar10_classifier_no=10
        inputs = keras.Input((32, 32, 3))
        conv_base = self.get_convbase(inputs)
        angle_classifier = self.get_classifier(conv_base, angle_classifier_no, "angle",count=15)
        cifar10_classifier = self.get_classifier(conv_base, cifar10_classifier_no, "cifar10",count=22)
        self.model = Model(
            inputs=inputs, 
            outputs=[cifar10_classifier, angle_classifier]
        )
        tf.keras.utils.plot_model(self.model, to_file='final_mtl_model.png', show_shapes=True,show_layer_names=True)
    
   
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
        self.model.load_weights('./custom_mtl_model.h5')
        
    def execute_predict(self,input_data):
        st1=time.perf_counter()
        out=self.model.predict(input_data)
        et1=time.perf_counter()
        el1=et1-st1
        print(f'Elapsed Time: {el1}')
        return out
    
    def print_lays(self):
        for lay in self.model.layers:
            self.layer_list.append(lay)
            print(lay)
        print(f'Total number of Layers : {len(self.layer_list)}')
    def execute_lbl(self,input_data):
        st2=time.perf_counter()
        out=buffer=input_data
        for idx in range(1,len(self.model.layers)):
            
            if idx <= 15:
                print(f'Executing index {idx} : {self.model.layers[idx]}')
                out=buffer=self.model.layers[idx](out)
            elif idx in [16,18,20,22,24,26,28]:
                print(f'Executing index {idx} : {self.model.layers[idx]}')
                out=self.model.layers[idx](out)
            elif idx in [17,19,21,23,25,27,29]:
                print(f'Executing index {idx} : {self.model.layers[idx]}')
                buffer=self.model.layers[idx](buffer)
        et2=time.perf_counter()
        el2=et2-st2
        print(f'Elapsed Time: {el2}')
        return out,buffer
    
    def gte_input_list(self,input_data):
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
    
    def execute_on_core(self,layer_id,input_data,dummy_data):
        dummy_data=dummy_data
        # print(self.layer_list[layer_id].name)
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out
        
    


# %%
def compute_execution_time(target_instance, target_method, core_id=0, *args):
    try:
        psutil.Process().cpu_affinity([core_id])
    except AttributeError:
        pass  
    start_time = time.perf_counter()
    tt=getattr(target_instance, target_method)(*args)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    # print(f"Execution time on core {core_id}: {execution_time} seconds")
    return execution_time,tt

def compute_pair_execution_time(target_instance, target_method, core_id=[0,0], *args):
    
    st1=time.perf_counter()
    try:
        psutil.Process().cpu_affinity([core_id[0]])
    except AttributeError:
        pass  
    et1=time.perf_counter()
    layer=args[0]
    inp_seq=args[1]
    st2 = time.perf_counter()
    tt=getattr(target_instance, target_method)(layer[0],inp_seq[0],'dum')
    et2 = time.perf_counter()
    
    st3=time.perf_counter()
    try:
        psutil.Process().cpu_affinity([core_id[1]])
    except AttributeError:
        pass
    
    et3=time.perf_counter()
    st4=time.perf_counter()
    tt2=getattr(target_instance, target_method)(layer[1],inp_seq[1],tt)
    et4 = time.perf_counter()
    
    el1=et4-st1
    el2=et2-st2
    el3=et3-st3
    el4=et4-st4
    execution_time = el1+el2+el3+el4
    # print(f"Execution time on core {core_id}: {execution_time} seconds")
    return el1,tt2


# %%
NO_OF_LAYERS=30
NO_OF_CPU=24

# %%
def try_grid(obj,layer_ids,core_ids,input_data):
    # temp=[0]*2
    temp_out=input_data
    # st=time.perf_counter()
    # for lay in range(len(layer_ids)):
    temp,temp_out=compute_pair_execution_time(obj,'execute_on_core',core_ids,layer_ids,temp_out)  
        
    # et=time.perf_counter()
    # el=et-st
    return temp, temp
    

def perform_grid(obj,lays,inp_seq):
    res=np.zeros((NO_OF_CPU,NO_OF_CPU),dtype =  float)
    for i in range(NO_OF_CPU):
        for j in range(NO_OF_CPU):
            #Now schedule this function on the CPU-0 to run the two layers on the different CPUs
            # temp,res[i][j]= compute_execution_time_of_function(try_grid,0,obj,lays,[i,j],inp_seq)
            # st=time.perf_counter()
            res[i][j],temp=try_grid(obj,lays,[i,j],inp_seq)
            
            # et=time.perf_counter()
            # el=et-st
            # res[i][j]=el
        time.sleep(0.5)
    return res
 

# %%
def make_heatmap(readings,name):
    plt.imshow(readings, cmap='cividis', interpolation='nearest')
    path='./img/conn/'+name+'.png'
    plt.colorbar()
    plt.title(name)
    plt.savefig(path)
    plt.close()
    # plt.show()



def write_to_csv(name,res):
    csv_file_path=name
    row_headings =[str(i) for i in range(24)]
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row_headings)
        # Write each row of the array to the CSV file
        for row in res:
            
            csv_writer.writerow(row)


# %%
def do_profiling(obj,layers,inp_seq):

    NO_EXP=10
    main_readings=[]

    # tag=f'lay{conn+1}{conn+2}'

    for i in range(NO_EXP):
        res=perform_grid(obj,layers,inp_seq)
        main_readings.append(res)
        csv_name=f'./readings/conn/custom_mtl_lay_{layers[0]}_to_{layers[1]}_r{i+1}.csv'
        heat_map_name=f'heat_map_custom_mtl_lay_{layers[0]}_to_{layers[1]}_r{i+1}'
        write_to_csv(csv_name,res)
        make_heatmap(res,heat_map_name)
        gc.collect()
        time.sleep(0.5)
        gc.collect()
        
    
    result_ave = np.mean(main_readings, axis=0)
    result_ave

    avcsv_name=f'./readings/conn/ave_reads_lay_{layers[0]}_to_{layers[1]}.csv'

    write_to_csv(avcsv_name,result_ave)
    avf_name=f'ave_reads_lay{layers[0]}_to_{layers[1]}'
    make_heatmap(result_ave,avf_name)

# %%
conn_list=[[1,2],[2,3],[3,4],[4,5],[5,6],[15,16],[16,18],[18,20],[22,24],[24,26]]
# conn_list=[[15,16]]

# %%
image = np.random.rand(1,32,32,3)

# %%
obj=custom_MTL()
obj.load_weights()
# out=obj.model.predict(image)
obj.print_lays()
obj.execute_predict(image)

# %%
INPUT_LIST=obj.gte_input_list(image)
len(INPUT_LIST)

# %%
# for i in range(1,len(INPUT_LIST)):
out=obj.execute_on_core (28,INPUT_LIST[28],'45')

# %%
for ele in conn_list:
    print(f'PROFILING for Layer {ele}')
    layers=[ele[0],ele[1]]
    inp_da=[INPUT_LIST[ele[0]],INPUT_LIST[ele[1]]]
    do_profiling(obj=obj,layers=layers,inp_seq=inp_da)
    print('Sleeping for 4 seconds')
    gc.collect()
    time.sleep(4)


# %%
def calculate_mean(num_list):

    if not num_list:
        return None  # Return None if the list is empty

    total = sum(num_list)  # Calculate the sum of all numbers in the list
    mean = total / len(num_list)  # Calculate the mean by dividing the sum by the number of elements
    return mean


# %%



