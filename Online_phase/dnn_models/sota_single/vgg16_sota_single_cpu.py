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
    def __init__(self):
        self.weight_set=False
        self.partition_done=False
        self.input_loaded=False
        self.layer_list=[]
        
        
        self.model = Sequential()
        
        # Block 1
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # Block 2
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # Block 3
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # Block 4
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # Block 5
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        # Flatten
        self.model.add(Flatten())
        
        # Fully connected layers
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(1000, activation='softmax'))  # Output layer with 1000 classes for ImageNet
        # tf.keras.utils.plot_model(self.model, to_file='vgg16_model.png', show_shapes=True)
        self.NO_OF_LAYERS= len(self.model.layers)
        self.loadWeights()
        self.make_partition()
        
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
        return self.output,elt1,0,0,[] 
        # return elt1,self.output
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
    def execute_layer_by_layer_sample_inp_base(self,inp):

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
    
    def get_best_cpu(self,cand_layer,cpu_log):
        # print(f"***********Current CPU_LOG {cpu_log}")
        
        # best_map=[6, 9, 9, 10, 10, 11, 12, 16, 1] # start_with_best
        # best_map=[0, 9, 9, 10, 10, 11, 12, 16, 1] # start_with_cpu_0
        # best_map=[14, 2, 2, 1, 1, 13, 13, 17, 1] # start_with_second_best_cpu
        # best_map=[5, 5, 10, 10, 10, 11, 12, 16, 1] # start_with_fifth_best_cpu
        best_map = [1,22,2,21,3,20,4,19,5] # just random map
        while True:
            
            if len(cpu_log)<=0:
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
            else:
                # try:
                # # my_list = cnn_prof[up_obj.curent_layer][up_obj.prev_cpu]
                found_element = None
                # for element in my_list:
                #     if element in cpu_log:
                #         found_element = element
                #         break
                if best_map[cand_layer] in cpu_log:
                    found_element=best_map[cand_layer]
                    cpu_log.remove(found_element)
                    # print(f"CPU assigned")
                    break
                else:
                    print(f'________Waiting for best CPU for layer {cand_layer}__________')
                    continue
                

                
                
        # print(f"Returning CPU")
        return found_element, cpu_log

    
    def execute_layer_by_layer_sample(self,inp):
        self.executed_map=[]
        # print('Helo')
        t1=time.perf_counter()
        temp_out=inp
        layer_execution_time=0
        select_cpu_time=0
        for lay in range(self.NO_OF_LAYERS):
            # print('In layer loop')
            t31=time.perf_counter()
            # best_proc_ele,cpu_log= self.get_best_cpu(lay,cpu_log)
            # print(f"CPU Returned")
            # p=psutil.Process()
            # print(f"################Changing the affinity from CPU {p.cpu_affinity()} to CPU NA for executing layer {lay}#####",flush=True)
            # p.cpu_affinity([best_proc_ele])
            # self.executed_map.append(best_proc_ele)
            self.load_layer(lay)
            # print(f'Layer and Weights loaded for CNN-9 layer ID : {lay} succesfully',flush=True)
            t12=time.perf_counter()
            temp_out=self.layer(temp_out)
            t22=time.perf_counter()
            e2=t22-t12
            e3=t12-t31
            layer_execution_time +=e2
            select_cpu_time += e3
            # cpu_log.append(best_proc_ele)
        temp_out=temp_out.numpy()
        t2=time.perf_counter()
        elapsed_time=t2-t1
        return temp_out,elapsed_time,layer_execution_time,select_cpu_time,self.executed_map
               
    
    def execute_on_core(self,layer_id,input_data):
                
        self.temp_out=self.layer_list[layer_id](input_data)
        
        return self.temp_out



       
# ====================================VGG16 End =================================================================



def child(worker: int,cpu_log,mode:int,input_data) -> None:
    
    cp1=time.perf_counter()
    p = psutil.Process()
    
    p.cpu_affinity([worker])
    # print(f'Type of CPU_LOG {type(cpu_log)}', flush=True)
    # print(f'Current CPU_LOG : {cpu_log}',flush=True)
    
    print(f"Child #{worker}: Starting CPU intensive task now for 4 seconds on {p.cpu_affinity()}...", flush=True)
    
    obj=None
    cp2=time.perf_counter()
    if mode==22:
        obj=vgg16_in()
    elif mode == 7:
        obj=vgg16_in()
        r1=obj.make_partition()
    # print("==========================object created")
    cp4=time.perf_counter()
    
    # r2=obj.execute_on_core(layer_id=layer_id, input_data=input_data, proc_ele=worker)
    cpu_log.append(worker)
    # print(f'I am after appending worker')
    r2=obj.execute_full_network_sample(imp=input_data)
    # r2=obj.execute_layer_by_layer_sample(inp=input_data)
    # r2=obj.execute_layer_by_layer_sample_inp_base(inp=input_data)

    cp5=time.perf_counter()
    
    el1=cp2-cp1
    el2=cp4-cp2
    # el2=el2+r2[2]
    print(f"Child #{worker}: Finished CPU intensive task on {p.cpu_affinity()} for CNN-{mode} and takes affinity time : {el1} and creating object loading takes: {el2} ", flush=True)
    p.cpu_affinity([])
    #r2[0] = model output
    # r2[1] = elapsed time
    # r[2] = layer_ex time
    # r[3] =  CPU delay
    # r[4] = executed CPU map
    # el2= contructor latency
    # cpu_log = current available CPUs
    return r2[0], r2[1],r2[2],r2[3],r2[4],el2,cpu_log

    
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
        self.cpu_track=token[4]
        self.obj_time = token[5]
        
        print(f"CNN-{self.mode} is completed succesfully with total execution time of : {self.exe_time} and object creation and loading time of {self.obj_time}", flush=True)
        print(f'CPU TRACK for Model : {self.cpu_track}')
        self.total_inf_time= self.exe_time + self.obj_time
        self.write2csv([self.exe_time,self.cpu_delay,self.cpu_delay,self.obj_time,self.total_inf_time,self.cpu_track])
    
    
def get_best_cpu(up_obj:update_class,cpu_log):
    # print(f"***********Current CPU_LOG {cpu_log}")
    
    best_map=[6, 9, 9, 10, 10, 11, 12, 16, 1] # start_with_best
    # best_map=[0, 9, 9, 10, 10, 11, 12, 16, 1] # start_with_cpu_0
    # best_map=[14, 2, 2, 1, 1, 13, 13, 17, 1] # start_with_second_best_cpu
    # best_map=[5, 5, 10, 10, 10, 11, 12, 16, 1] # start_with_fifth_best_cpu
    while True:
        
        if len(cpu_log)<=0:
            print('__________________________Waiting for CPU to Get Free_________________')
            continue
        else:
            # try:
            # # my_list = cnn_prof[up_obj.curent_layer][up_obj.prev_cpu]
            found_element = None
            # for element in my_list:
            #     if element in cpu_log:
            #         found_element = element
            #         break
            if best_map[up_obj.curent_layer] in cpu_log:
                found_element=best_map[up_obj.curent_layer]
                cpu_log.remove(found_element)
                break
            else:
                print(f'________Waiting for best CPU__________')
                continue
                       
    
    return found_element, cpu_log


def write2csv(file_name,write_token):
        with open(file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(write_token)

def main(para=1) -> None:

    np.random.seed(75)

    NO_OF_CPU=24
    NO_OF_INFERENCE=para
    updators=[None]*NO_OF_INFERENCE
    inputs=[None]*NO_OF_INFERENCE
    
    main_et_file=f'./readings/sota/predict_single/inst_set_{NO_OF_INFERENCE}_delay/main_ET_random.csv'
    # cpu_log=[i for i in range(24)]
    pool=mp.Pool()
    manager= mp.Manager()
    cpu_log= manager.list([i for i in range(NO_OF_CPU)] )
    workers: int = pool._processes
    print(f"Running pool with {workers} workers",flush=True)
    
    # with open('vgg16_greedy_profile.pkl', 'rb') as file:
    #     cnn9_profile = pickle.load(file)
    
    def sleep_random_time(min_seconds, max_seconds):
        sleep_time = random.uniform(min_seconds, max_seconds)
        print(F'############## SLEEPING FOR {sleep_time} #############')
        time.sleep(sleep_time)
        
    min_sleep_time = 0.19
    max_sleep_time = 0.39
    mr1=time.perf_counter()
    print(f"Main programe starting time reading of perf_counter : {mr1}",flush=True)
    for inst in range(NO_OF_INFERENCE):
        sleep_random_time(min_sleep_time, max_sleep_time)
        updators[inst]=update_class(mode=22,read_file_name=f'./readings/sota/predict_single/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv')
        inputs[inst]=np.random.rand(1, 224, 224, 3).astype(np.float32)
        sing_cp=inst%NO_OF_CPU
        pool.apply_async(child, (sing_cp,cpu_log,22,inputs[inst]),callback=updators[inst].updater)
    
    
    # Wait for children to finnish
    pool.close()
    pool.join()
    mr2=time.perf_counter()
    met=mr2-mr1
    write2csv(main_et_file,[met])
    print(f"Main programe end at {mr2} taking {met} seconds",flush=True)
    # pass

import gc
import sys
if __name__ == '__main__':
    para=int(sys.argv[1])
    for rr in range(10):
        gc.collect()
        time.sleep(1)
        main(para=para)
        gc.collect()