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
    def __init__(self,custom_mtl_profile) -> None:
        
        self.layer_list=[]
        self.weight_set=False
        self.partition_done = False
        self.NO_OF_LAYERS=30
        self.prof_data=custom_mtl_profile
        # x_shape, y_angle_shape, y_cifar10_shape=(50000, 32, 32, 3), (50000, 4), (50000, 10)
        # angle_classifier_no=4
        # cifar10_classifier_no=10
        # inputs = keras.Input((32, 32, 3))
        # conv_base = self.get_convbase(inputs)
        # angle_classifier = self.get_classifier(conv_base, angle_classifier_no, "angle",count=15)
        # cifar10_classifier = self.get_classifier(conv_base, cifar10_classifier_no, "cifar10",count=22)
        # self.model = Model(
        #     inputs=inputs, 
        #     outputs=[cifar10_classifier, angle_classifier]
        # )
        # self.load_weights()
        # self.make_partition()
        # # tf.keras.utils.plot_model(self.model, to_file='final_mtl_model.png', show_shapes=True,show_layer_names=True)
    
   
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
        self.dir_path='../pickle_layers'
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
        
    def execute_full(self,input_data):
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
        print(f'Elapsed Time: {el2}')
        return out,buffer
    
    def get_best_cpu(self,cand_layer,cpu_log,last_cpu):
        # print(f"***********Current CPU_LOG {cpu_log}")
        
        while True:
            if not any(cpu_log):
                print('__________________________Waiting for CPU to Get Free_________________')
                continue
            else:
                if cand_layer == 0 or cand_layer ==1:
                    my_list = self.prof_data[cand_layer][0]
                else:
                    my_list = self.prof_data[cand_layer][last_cpu]
                found_element = None
                for element in my_list:
                    if element :
                        found_element = element
                        cpu_log[found_element]=False
                        break
                
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
    
    
    def execute_layer_by_layer_sample(self,input_data,cpu_log):
        # print('--------In the Function--------------')
        t1=time.perf_counter()
        self.executed_map=[]
        best_proc_ele=0
        p=psutil.Process()
        out=buffer=input_data
        layer_execution_time=0
        select_cpu_time=0
        layer_loading_time=0
        for idx in range(self.NO_OF_LAYERS):
            # print('---------In LOOP---------------')
            curr_lay=None
            t31=time.perf_counter()
            
            best_proc_ele,cpu_log= self.get_naive_best_cpu(idx,cpu_log)
            # print("Get the cPU")
            p.cpu_affinity([best_proc_ele])
            # print('Appended')
            self.executed_map.append(best_proc_ele)
            t41=time.perf_counter()
            
            curr_lay=self.load_layer(idx)
            # print('LAyers are Loaded')
            t12=time.perf_counter()
            if idx ==0:
                # out=curr_lay(out)
                out=out
            elif idx <= 15:
                # print('layer ging to execute')
                # print(curr_lay)
                # try:
                out=curr_lay(out)
                buffer=out
                # except Exception as e:
                #     # Handle the exception
                #     print("An error occurred:", e)
                # print(f'Executing index {idx} ')
            elif idx in [16,18,20,22,24,26,28]:
                out=curr_lay(out)
                # print(f'Executing index {idx}')

            elif idx in [17,19,21,23,25,27,29]:
                buffer=curr_lay(buffer)
            #     print(f'Executing index {idx} ')
            # print("After the if ELse condition")
            t22=time.perf_counter()
            curr_lay=None
            e2=t22-t12
            e3=t12-t31
            e4=t12-t41
            layer_execution_time +=e2
            select_cpu_time += e3
            layer_loading_time +=e4
            cpu_log[best_proc_ele]=True
        t2=time.perf_counter()
        elapsed_time=t2-t1
        # print(f'Elapsed Time: {el2}')
        return (out,buffer),elapsed_time,layer_execution_time,select_cpu_time,layer_loading_time,self.executed_map
    
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

def child(worker: int,cpu_log,mode:int,input_data,DNN_profile) -> None:
    
    cp1=time.perf_counter()
    p = psutil.Process()
    
    # p.cpu_affinity([2])
    # print(f'Type of CPU_LOG {type(cpu_log)}', flush=True)
    # print(f'Current CPU_LOG : {cpu_log}',flush=True)
    
    print(f"Child #{worker}: Starting CPU intensive task now for 4 seconds on {p.cpu_affinity()}...", flush=True)
    
    obj=None
    cp2=time.perf_counter()
    if mode==30:
        obj=custom_MTL(custom_mtl_profile=DNN_profile)
    elif mode == 30:
        obj=custom_MTL(custom_mtl_profile=DNN_profile)
        # r1=obj.make_partition()
    
    cp4=time.perf_counter()
    
    # r2=obj.execute_on_core(layer_id=layer_id, input_data=input_data, proc_ele=worker)
    # cpu_log.append(worker)
    print(f'------Created Object------------------')
    # r2=obj.execute_full_network_sample(imp=input_data)
    # r2=obj.execute_full(imp=input_data)
    # r2=obj.execute_layer_by_layer_sample(inp=input_data,cpu_log=cpu_log)
    r2=obj.execute_layer_by_layer_sample(input_data=input_data,cpu_log=cpu_log)

    cp5=time.perf_counter()
    
    el1=cp2-cp1
    el2=cp4-cp2
    el3=cp5-cp2
    # el2=el2+r2[2]
    print(f"Child #{worker}: Finished CPU intensive task on {p.cpu_affinity()} for CNN-{mode} and takes affinity time : {el1} and creating object loading takes: {el2} TOTAL: {el3} ", flush=True)
    #r2[0] = model output
    # r2[1] = elapsed time
    # r[2] = layer_ex time
    # r[3] =  CPU delay
    # r2[4] = layer_load_time
    # r[5] = executed CPU map
    # el2= contructor latency
    # cpu_log = current available CPUs
    return r2[0], r2[1],r2[2],r2[3],r2[4],r2[5],el2,el3,cpu_log


# ===================================================Multiprocessing Worker Process End ===========================================


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
        self.totl=token[7]
        print(f"CNN-{self.mode} is completed succesfully with total execution time of : {self.exe_time} and object creation and loading time of {self.obj_time}", flush=True)
        print(f'CPU TRACK for Model : {self.cpu_track}')
        self.total_inf_time= self.exe_time + self.obj_time
        self.write2csv([self.exe_time,self.cpu_delay,self.layer_loading_time,self.obj_time,self.total_inf_time,self.cpu_track,self.totl])
        print("CSV file UPDATED")


def main(para=1) -> None:

    np.random.seed(18)

    NO_OF_CPU=24
    NO_OF_LAYERS=30
    NO_OF_INFERENCE=para
    updators=[None]*NO_OF_INFERENCE
    inputs=[None]*NO_OF_INFERENCE
    
    main_et_file=f'./readings/naive_cpu/inst_set_{NO_OF_INFERENCE}_delay/main_ET_random.csv'
    # cpu_log=[i for i in range(24)]
    pool=mp.Pool()
    manager= mp.Manager()
    # log_cpu=[True]*NO_OF_CPU
    cpu_log= manager.list([True]*NO_OF_CPU)
    workers: int = pool._processes
    print(f"Running pool with {workers} workers",flush=True)


   

    # vgg_weights=manager.list(main_weight_list)
    with open('./custom_mtl_naive_cpuProfiling_mappings.pkl', 'rb') as file:
        custom_mtl_profile = pickle.load(file)
    
    def sleep_random_time(min_seconds, max_seconds):
        sleep_time = random.uniform(min_seconds, max_seconds)
        print(F'############## SLEEPING FOR {sleep_time} #############')
        time.sleep(sleep_time)
        
    min_sleep_time = 0.19
    max_sleep_time = 0.39
    
    
    mr1=time.perf_counter()
    print(f"Main programe starting time reading of perf_counter : {mr1}",flush=True)
    
    for inst in range(NO_OF_INFERENCE):
        # sleep_random_time(min_sleep_time, max_sleep_time)
        updators[inst]=update_class(mode=30,read_file_name=f'./readings/naive_cpu/inst_set_{NO_OF_INFERENCE}_delay/inf{inst}_greedy.csv')
        inputs[inst]=np.random.rand(1, 32, 32, 3).astype(np.float32)
        pool.apply_async(child, (0,cpu_log,30,inputs[inst],custom_mtl_profile),callback=updators[inst].updater)
    
    
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
    gc.collect()
    for rr in range(1):
        gc.collect()
        # time.sleep(0.2)
        gc.collect()
        main(para=para)