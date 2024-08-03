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
    return execution_time,tt

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


# %%
NO_OF_CPU=24
NO_OF_LAYER=11

# %%

def try_grid(obj,layer_ids,core_ids,input_data):
    # temp=[0]*2
    temp_out=input_data[0]
    # st=time.perf_counter()
    # for lay in range(len(layer_ids)):
    temp,temp_out=compute_pair_execution_time(obj,'execute_on_core',core_ids,layer_ids[0],temp_out)  
        
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
    path='./img/all_conn/'+name+'.png'
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
        csv_name=f'./reads/all_conn/pilotnet_lay_{layers[0]}_to_{layers[1]}_r{i+1}.csv'
        heat_map_name=f'heat_map_pilotnet_lay_{layers[0]}_to_{layers[1]}_r{i+1}'
        write_to_csv(csv_name,res)
        make_heatmap(res,heat_map_name)
        
        
    result_ave = np.mean(main_readings, axis=0)
    result_ave

    avcsv_name=f'./reads/all_conn/ave_reads_lay_{layers[0]}_to_{layers[1]}.csv'

    write_to_csv(avcsv_name,result_ave)
    avf_name=f'ave_reads_lay{layers[0]}_to_{layers[1]}'
    make_heatmap(result_ave,avf_name)

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
    
len(INPUT_LIST[2][0])

# %%
lays=[13,14]
inp_seq=INPUT_LIST[13:15]

# %%
conn_list=[0,1,2,3,4,5,6,7,8,9]
# conn_list=[20]
# obj.model.summary()

# %%


# %%
for ele in conn_list:
    print(f'PROFILING for Layer {ele}')
    layers=[ele,ele+1]
    inp_da=[INPUT_LIST[ele],INPUT_LIST[ele+1]]
    do_profiling(obj=obj,layers=layers,inp_seq=inp_da)
    print('Sleeping for 4 seconds')
    time.sleep(4)


