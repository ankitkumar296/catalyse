README

To get the results from the CATALYSE first you need to profile the DNNs on the target hardware by system details i.e. number of PEs available etc.
It follows the implementation and experimental evaluation from the submitted work CATALYSE :Communication-aware Task Mapping for DNNs on Multicore Systems at Edge
it has two modules offline_phase and online_phase
Follow the below procedure

Anonymous Repository:

CATALYSE
->Online_phase
->Offline_phase


*** REQUIRMENTS: ****
Linux OS: ubuntu 22.04 LTS
tensorflow 
python 3.11
matplotlib
numpy
pandas
psutil
***************************


*** Hardware used ***
CPU:AMD Ryzen 9 5900X
12 core, 24 threads(considers as PE)
RAM: 64 GB
*********************


============================OFFLINE PHASE:==============================
Offline Phse directory Structure
->customMTL
->pilotney
->self_driving_mtl
->squeeznet
->vgg16


\\in offline phase the profiling tables are created for the target hardware according to the Algorithm 1,2 and 3
\\All these table are created using file placed in dnn_name\profiling 
\\It creates the percentage of latency spent on communication
\\PE priority table for DNN
\\Adjust the hardware info accordingly in code file i.e. number of PE available

**for creating the profiled priority tables **
-profile each layer for every available PE in the system (using file single_layer_profile_#dnn_name#.py)
-profile consecutive layers of each connection type for all available PEs (using file #dnn_name#_conn_profile.py)
-use helper1.ipynb file to get the table containing the percentage amount of latency spend by each connection between different PE connection
-use helper2.ipynb file to create the priority table using the data we get from profiling which is managed in csv files
-put these priority table created from ipynb helper files into the respective path 
	i.e. Online_phase\dnn_models\_profilings\greedy_profile\#dnn_name#_greedy_profile.kl
	i.e. Online_phase\dnn_models\_profilings\naive_profile\#dnn_name#_naive_cpu.kl
	i.e. Online_phase\dnn_models\_profilings\trans_data\#dnn_name#_data_trans.kl

**creating naive profiling(SOTA) table **
-use naive_CPU_profiling.ipynb and csv files for profiling each PE for each layer to create the naive profiling table named SOTA in paper

** creating the table for managing the size of output data flow throughout the DNN **
-this is fixed for each DNN so can be reused as it is from the provided artifact

Place all pkl files to there respective paths with appropriate file names as mentioned

NOTE:
For VGG16offline phase get model.h5 file and pikle layers from online phase
model.h5:CATALYSE\Online_phase\dnn_models\_model_files
pikle layers :CATALYSE\Online_phase\dnn_models\_dnn_saved_layers\vgg

Weigths can be downloaded via:https://drive.google.com/drive/folders/1verObPtHF-hX5zli1PuhuqiJ3lXGbE6N?usp=sharing

=================================================================================================





=================================ONLINE PHASE DIRECTORY STRUCTURE:==========================================
Online_phase
->dnn_models- directories containing the architectural code and implementation 

	->_dnn_saved_layers : saved trained layers of neural networks in the pkl format 
	->_model_files : model file in h5 format (saved model) 
	->_profilings : directory containing all the profiled details of all network on particular hardware, prepared in the OFFLINE phase
	->greedy_wc :greedy
	->naive_cpu : SOTA
	->random_cpu 
	->sota_all : ALL PE 
	->sota_single : SINGLE PE 
	->threshold_switching : CATALYSE

->readings- Directory for managing the obtained readings from different experiments, further divided on the basis of number of instances of NNs 

	->greedy_wc : Greedy
	->naïve_cpu : SOTA
	->random 
	->sota_all_cpu: ALL PE 
	->sota_single_cpu : SINGLE PE 
	->threshold_switching : CATALYSE
	-helper1.ipynb
	-helper2.ipynb

//Main files through which we can run the mixture of DNNs using different settings 
//provide number of inference as a argument in these files
//e.g python main_random.py 32

-main_greedy_wc.py : Greedy
-main_naïve_cpu.py : SOTA
-main_random.py  
-main_sota_all_cpu.py :All PE
-main_sota_single_cpu.py :Single PE
-main_threshold_switching.py: (CATALYSE) 




***** Executing the suit of 32 DNN inferences with different  settings *****

to execute the suit with any of the six method execute the respective main_## file
results will be saved in readings\#method#\inst_set_##_delay\inf#.csv 
use the helper#.ipynb to view the performance of the experiments
helper1.ipynb:to get the average minimum and maximum inference latency
helper2.ipynb: to get the percentage of deadline satisfied based on the threshold provided

********************************************************************************


