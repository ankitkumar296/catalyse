# %%
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

# %%
class squeeznet():
    def __init__(self,input_shape, nb_classes, use_bypass=False, dropout_rate=None, compression=1.0) -> None:
        
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
        input_img = Input(shape=input_shape)
        self.model=ker.Sequential
        x = Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
        
        x = self.create_fire_module(x, int(16*compression), name='fire2')
        x = self.create_fire_module(x, int(16*compression), name='fire3', use_bypass=use_bypass)
        x = self.create_fire_module(x, int(32*compression), name='fire4')
        
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
        
        x = self.create_fire_module(x, int(32*compression), name='fire5', use_bypass=use_bypass)
        x = self.create_fire_module(x, int(48*compression), name='fire6')
        x = self.create_fire_module(x, int(48*compression), name='fire7', use_bypass=use_bypass)
        x = self.create_fire_module(x, int(64*compression), name='fire8')
        
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool8')(x)
        
        x = self.create_fire_module(x, int(64*compression), name='fire9', use_bypass=use_bypass)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)
            
        x = self.output(x, nb_classes)
        
        self.model=Model(inputs=(input_img), outputs=x)

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
        pass
    
    def execute_predict(self, input_data):
        st1=time.perf_counter()
        out=self.model.predict(input_data)
        et1=time.perf_counter()
        el=et1-st1
        print(el)
        return out
    def print_layrs(self):
        i=0
        for lay in self.model.layers:
            self.layer_list.append(lay)
            print(f'Index: {i} --> {lay.name}')
            i+=1
    def execute_lbl(self, input_data):
        st2=time.perf_counter()
        self.buffer=[None,None,None,None]
        self.buffer[0]=input_data
        
        for idx in range(len(self.model.layers)):
            
            curr_lay=self.model.layers[idx]
            
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
        print(el2)
        return self.buffer[0].numpy()
    
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
NO_OF_LAYERS=44
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
        csv_name=f'./readings/conn/squeeznet_lay_{layers[0]}_to_{layers[1]}_r{i+1}.csv'
        heat_map_name=f'heat_map_squeeznet_lay_{layers[0]}_to_{layers[1]}_r{i+1}'
        write_to_csv(csv_name,res)
        make_heatmap(res,heat_map_name)
        # gc.collect()
        # time.sleep(0.5)
        # gc.collect()
        
    
    result_ave = np.mean(main_readings, axis=0)
    result_ave

    avcsv_name=f'./readings/conn/ave_reads_lay_{layers[0]}_to_{layers[1]}.csv'

    write_to_csv(avcsv_name,result_ave)
    avf_name=f'ave_reads_lay{layers[0]}_to_{layers[1]}'
    make_heatmap(result_ave,avf_name)

# %%
conn_list=[[1,2],[2,3],[3,4],[5,6],[6,7],[10,11],[11,12],[15,16],[35,40],[40,41],[42,43],]
# conn_list=[[35,40]]

# %%
obj=squeeznet(input_shape=(224,224,3),nb_classes=1000,use_bypass=True)
obj.load_weights()
obj.print_layrs()
images = np.random.rand(1,224,224,3)
INPUT_LIST=obj.get_input_list(images)

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
obj.print_layrs()
print(len(INPUT_LIST))

for i in range(1,len(INPUT_LIST)):
    print(f"Executing {i}")
    out=obj.execute_on_core(i,INPUT_LIST[i],'jij')

# %%



