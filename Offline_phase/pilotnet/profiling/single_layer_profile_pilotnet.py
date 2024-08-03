# %%

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import psutil
import time
import csv
import matplotlib.pyplot as plt



# %%

class pilotnet():
    def __init__(self) -> None:
        self.weight_set=False
        self.partition_done=False
        self.input_loaded=False
        self.layer_list=[]
        
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
        return elt1
    
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
    
    def execute_layer_by_layer_sample(self,inp):
        temp_out=inp
        for lay in self.layer_list:
            
            temp_out=lay(temp_out)
               
        temp_out=np.array(temp_out)
                    
        return temp_out
    
    def execute_on_core(self,layer_id,input_data):
                
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
    next_layer=args[0]
    st2 = time.perf_counter()
    tt=getattr(target_instance, target_method)(*args)
    et2 = time.perf_counter()
    
    st3=time.perf_counter()
    try:
        psutil.Process().cpu_affinity([core_id[1]])
    except AttributeError:
        pass
    
    et3=time.perf_counter()
    st4=time.perf_counter()
    tt2=getattr(target_instance, target_method)(next_layer+1,tt)
    et4 = time.perf_counter()
    
    el1=et4-st1
    el2=et2-st2
    el3=et3-st3
    el4=et4-st4
    execution_time = el1+el2+el3+el4
    # print(f"Execution time on core {core_id}: {execution_time} seconds")
    return el1,tt2

def give_mean(data):
    if not data:
        return None  # Return None if the list is empty
    
    total = sum(data)
    average = total / len(data)
    return average

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
NO_OF_CPU=24
NO_OF_LAYER=11

# %%
obj =pilotnet()
obj.loadWeights()
obj.make_partition()
obj.execute_full_network()
obj.execute_full_partition()


# %%
random_input = np.random.rand(1, 66, 200, 3).astype(np.float32)
INPUT_LIST=[]
INPUT_LIST.append(random_input)
for i in range(1,NO_OF_LAYER):
    tmp=obj.execute_on_core(i-1,INPUT_LIST[i-1])
    INPUT_LIST.append(tmp)

# %%
EXP=25

layers_ex=[]
for l in range(NO_OF_LAYER):
    cpu_ex=[]
    for cpu in range(NO_OF_CPU):
        temp=[]
        for i in range(EXP):
            temp.append(compute_execution_time(obj,'execute_on_core',cpu,l,INPUT_LIST[l]))
        
        temp=give_mean(temp)
        # temp=min(temp)
        cpu_ex.append(temp)
    
    layers_ex.append(cpu_ex)


save_2d_list_to_csv(layers_ex, "single_layer_average_profile_pilotnet_for_percentage.csv")


# %%
# layers_ex


