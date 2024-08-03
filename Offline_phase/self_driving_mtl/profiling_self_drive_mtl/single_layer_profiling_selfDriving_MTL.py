# %%
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import psutil
import csv
import gc

# %%
class self_driving_mtl():
    def __init__(self) -> None:
            
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
    
    def print_layers(self):
        # self.layer_list=[]
        
        for lay in self.model.layers:
            print(lay.name)
            # self.layer_list.append(lay)
    
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
    
    def execute_on_core(self,layer_id,input_data):
        # dummy_data=dummy_data
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
    return execution_time

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
NO_OF_LAYERS=18
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
def save_2d_list_to_csv(data, file_name):
    """
    Save a 2D list to a CSV file.

    Parameters:
        data (list): The 2D list to be saved.
        file_name (str): The name of the CSV file to be created.
    """
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

    print(f'CSV file "{file_name}" has been created.')

# %%
def calculate_mean(num_list):

    # if not num_list:
    #     return None  # Return None if the list is empty

    total = sum(num_list)  # Calculate the sum of all numbers in the list
    mean = total / len(num_list)  # Calculate the mean by dividing the sum by the number of elements
    return mean


# %%
image= np.random.rand(1, 140,208,3)

# %%
obj=self_driving_mtl()
obj.load_weights()
obj.make_partition()
# obj.print_layers()
# obj.save_pickeled_layers()
obj.execute_lbl(image)

# %%
INPUT_LIST=obj.get_input_list(image)
len(INPUT_LIST)


# %%
for j in range(1,18):
    out=obj.execute_on_core (j,INPUT_LIST[j])
print(out)

# %%
EXP=25
dummy=[0]*NO_OF_CPU
layers_ex=[dummy]
layers_ex_ave=[dummy]
for l in range(1,NO_OF_LAYERS):
    cpu_ex=[]
    cpu_ex_avg=[]
    for cpu in range(NO_OF_CPU):
        temp=[]
        for i in range(EXP):
            temp.append(compute_execution_time(obj,'execute_on_core',cpu,l,INPUT_LIST[l]))
        
        temp1=calculate_mean(temp)
        temp2=min(temp)
        cpu_ex.append(temp2)
        cpu_ex_avg.append(temp1)
    time.sleep(0.5)
    layers_ex.append(cpu_ex)
    layers_ex_ave.append(cpu_ex_avg)


save_2d_list_to_csv(layers_ex, "./readings/single_layer_profile_self_driving_mtl_for_percentage.csv")
save_2d_list_to_csv(layers_ex_ave, "./readings/single_layer_ave_profile_self_driving_mtl_for_percentage.csv")



