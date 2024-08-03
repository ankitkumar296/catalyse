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

# ============================================= Network Start ============================================

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
        self.weight_set=False
        self.partition_done = False
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
        self.dir_path='pickled_layers'
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
    def execute_layer_by_layer_sample_inp(self, input_data):
        st2=time.perf_counter()
        self.buffer=[None,None,None,None]
        self.buffer[0]=input_data
        
        for idx in range(len(self.model.layers)):
            
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
    
    def execute_lbl_serial(self, input_data):
        st2=time.perf_counter()
        self.buffer=[None,None,None,None]
        self.buffer[0]=input_data
        
        for idx in range(len(self.model.layers)):
            
            curr_lay=self.load_layer(idx)
            print(f"Executing Layer -> {idx}")
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

# ++++++++++++++++++++++++++++++++++++++++++++ Network End +++++++++++++++++++++++++++++++++

# ===================================== Multi processing Child ==============================

def child(worker: int,cpu_log,mode:int,input_data) -> None:
    
    cp1=time.perf_counter()
    p = psutil.Process()
    # print('Enter the child')
    # p.cpu_affinity([worker])
    # print(f'Type of CPU_LOG {type(cpu_log)}', flush=True)
    # print(f'Current CPU_LOG : {cpu_log}',flush=True)
    
    print(f"Child #{worker}: Starting CPU intensive task now for 4 seconds on {p.cpu_affinity()}...", flush=True)
    
    # obj=None
    cp2=time.perf_counter()
    if mode==44:
        obj=squeeznet(input_shape=(224,224,3),nb_classes=1000,use_bypass=True)
    elif mode == 7:
        obj=squeeznet(input_shape=(224,224,3),nb_classes=1000,use_bypass=True)
        # r1=obj.make_partition()
    print('object_created')
    cp4=time.perf_counter()
    
    # r2=obj.execute_on_core(layer_id=layer_id, input_data=input_data, proc_ele=worker)
    # cpu_log.append(worker)
    # print(f'I am after appending worker')
    # r2=obj.execute_full_network_sample(imp=input_data)
    # r2=obj.execute_full(input_data=input_data)
    # r2=obj.execute_layer_by_layer_sample(inp=input_data,cpu_log=cpu_log)
    r2=obj.execute_layer_by_layer_sample_inp(input_data=input_data)

    cp5=time.perf_counter()
    
    el1=cp2-cp1
    el2=cp4-cp2
    el3=cp5-cp2
    # el2=el2+r2[2]
    print(f"Child #{worker}: Finished CPU intensive task on {p.cpu_affinity()} for CNN-{mode} and takes affinity time : {el1} and creating object loading takes: {el2} TOTAL: {el3} ", flush=True)
    # p.cpu_affinity([])
    #r2[0] = model output
    # r2[1] = elapsed time
    # r[2] = layer_ex time
    # r[3] =  CPU delay
    # r2[4] = layer_load_time
    # r[5] = executed CPU map
    # el2= contructor latency
    # cpu_log = current available CPUs
    return r2[0], r2[1],r2[2],r2[3],r2[4],r2[5],el2


# ====================================== Child Func End=====================================



def write2csv(file_name,write_token):
        with open(file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(write_token)

class update_class():
    def __init__(self,mode=0,read_file_name='read.csv') -> None:
        self.inter_result=None
        self.curent_layer=0
        self.lock=False
        self.mode=mode
        self.time_reads=[]
        self.exe_time=0
        self.obj_time=0
        self.prev_cpu=0
        self.cpu_track=[]
        self.read_file_name = read_file_name
        # self.pool_obj=pool_obj
        
        
    #writing readings to csv file
    def write2csv(self,write_token):
        with open(self.read_file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(write_token)
    def updater(self,token):
        self.inter_result=token[0]
        self.exe_time=token[1]
        self.layer_ex_time =token[2]
        self.cpu_delay= token[3]
        self.layer_loading_time=token[4]
        self.cpu_track=token[5]
        self.obj_time = token[6]
        
        print(f"CNN-{self.mode} is completed succesfully with total execution time of : {self.exe_time} and object creation and loading time of {self.obj_time}", flush=True)
        print(f'CPU TRACK for Model : {self.cpu_track}')
        self.total_inf_time= self.exe_time + self.obj_time
        self.write2csv([self.layer_ex_time,self.cpu_delay,self.layer_loading_time,self.obj_time,self.total_inf_time,self.cpu_track])
        print("CSV file UPDATED")

def main() -> None:

    np.random.seed(18)

    NO_OF_CPU=24
    NO_OF_LAYERS=44
    main_et_file='./readings/sota/lbl_all/_main_ET_greedy.csv'
    # cpu_log=[i for i in range(24)]
    pool=mp.Pool()
    manager= mp.Manager()
    cpu_log= manager.list([i for i in range(NO_OF_CPU)] )
    workers: int = pool._processes
    print(f"Running pool with {workers} workers",flush=True)


    # main_weight_list=[]

    # for layer_id in range(NO_OF_LAYERS):
        
    #     fname=f'./pilotnet_pickle/pilotnet_layer_{layer_id}.pkl'
    #     with open(fname, 'rb') as f:
    #         layer_config = pickle.load(f)
            
    #     main_weight_list.append(layer_config)
        
        # if layer_id in [2,5,9,13,17,18]:
        #     main_weight_list[layer_id] = None
        #     continue

        # weight=np.load(f'./vgg16_weights/layer_{layer_id}w.npy')
        # bias=np.load(f'./vgg16_weights/layer_{layer_id}b.npy')
        # main_weight_list[layer_id]=[weight,bias]

    # vgg_weights=manager.list(main_weight_list)
    # with open('pilotnet_vgg_greedy_profile_full.pkl', 'rb') as file:
    #     custom_mtl_profile = pickle.load(file)
    
    
    mr1=time.perf_counter()
    print(f"Main programe starting time reading of perf_counter : {mr1}",flush=True)
    up1=update_class(mode=44,read_file_name='./readings/sota/lbl_all/_inf1_greedy.csv')
    up2=update_class(mode=44,read_file_name='./readings/sota/lbl_all/_inf2_greedy.csv')
    # up3=update_class(mode=9)
    # up4=update_class(mode=9)
    inp1 = np.random.rand(1, 224, 224, 3).astype(np.float32)
    inp2 = np.random.rand(1, 224, 224, 3).astype(np.float32)
    # inp3 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    # inp4 = np.random.rand(1, 32, 32, 3).astype(np.float32)
    
    cpu=0

    # cpu , cpu_log= get_best_cpu(up1,cpu_log)
    # cpu_log.remove(0)
    pool.apply_async(child, (1,cpu_log,44,inp1),callback=up1.updater)
    # cpu , cpu_log= get_best_cpu(up2,cpu_log)
    # cpu_log.remove(1)
    pool.apply_async(child, (5,cpu_log,44,inp2),callback=up2.updater)
    # cpu , cpu_log= get_best_cpu(up3,cpu_log,cnn9_profile)
    # pool.apply_async(child, (cpu,cpu_log,9,inp3,up3.curent_layer),callback=up3.updater)
    # cpu , cpu_log= get_best_cpu(up4,cpu_log,cnn9_profile)
    # pool.apply_async(child, (cpu,cpu_log,9,inp4,up4.curent_layer),callback=up4.updater)

    
    # Wait for children to finnish
    pool.close()
    pool.join()
    mr2=time.perf_counter()
    met=mr2-mr1
    write2csv(main_et_file,[met])
    print(f"Main programe end at {mr2} taking {met} seconds",flush=True)
    # pass




import gc
if __name__ == '__main__':
    for i in range(30):
        gc.collect()
        gc.collect()
        time.sleep(0.2)
        main()
        gc.collect()