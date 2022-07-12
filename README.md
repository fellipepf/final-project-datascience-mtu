

<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>Deep learning and machine learning using actigraphy signals for schizophrenia classification</strong>  
    
---  
  ## Table of contents
1. [Introduction](#Introduction)  
2. [Software requirements](#software-requirements)  
3. [Software build](#software-build)  
4. [Project files description](#project-files-description)
5. [Dataset details](#dataset-details)
6. [Authors](#Authors)
7. [References](#References)  
 
---  
## 1. Introduction
This study aims to identify the presence of schizophrenia through the analysis of data collected through a wristwatch with a motor activities sensor using machine learning techniques.<br />
A research paper outlining this work is due to be presented at CBMS 2022 Conference, 21-23 July, Shenzhen, China.<br />
This builds on the work of Petter Jakobsen et al. [1]. 
  
## 2. Software requirements
Python 3.9  
  
  
## 3. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/fellipepf/final-project-datascience-mtu.git
 
```  
  
## 4. Project files description
  
* Feature Engineering and EDA

| File                                                                                | Description |    
|-------------------------------------------------------------------------------------|---|        
| [psykose_feature_engineering.py](./code/psykose_feature_engineering.py)             | Reads the raw files of control and patients and generate the features |
| [psykose_eda.py](./code/psykose_eda.py) <br/> [psykose_machine_learning_plots.py](./code/psykose_machine_learning_plots.py) ||
  
* Machine Learning 

| File                                                                            | Description                                                                   |    
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|                 
| [psykose_machine_learning_models.py](./code/psykose_machine_learning_models.py) | Contains the ML models for 10-Fold Cross-Validation and LOO using One Day Out |
| [psykose_machine_learning_loo.py](./code/psykose_machine_learning_loo.py)       | ML models for LOPO - leave one person out                                     |
| [hyperparameter_tuning.py](./code/hyperparameter_tuning.py)                                 | Perform the models hyper parameters tuning                                    |

- Deep Learning  

| File | Description |
|---|---|
| [psyche_2_loo_time_cat_cnn2d_6_5_noval_6t.py](./code/psyche_2_loo_time_cat_cnn2d_6_5_noval_6t.py) | Contains the Deep Learning model |


- Utils 

| File | Description                                |
|---|--------------------------------------------|
|[my_metrics.py](./code/my_metrics.py) | Class with the metrics used in the project |

## 5. Dataset details
Uses the publicly available Psykose dataset [2].

## 6. Authors
Fellipe Paes Ferreira, Aengus Daly 
## Contact details
Aengus Daly <br />
Munster Technological University, <br />
Cork City,<br />
Ireland

Email: aengus dot daly 'at' mtu.ie <br />
f dot paes-ferreira 'at' mycit dot ie

## 7. References

[1]  Petter Jakobsen et al., "PSYKOSE: A Motor Activity Database of Patients with Schizophrenia," 2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS), 2020, pp. 303-308, doi: 10.1109/CBMS49503.2020.00064.<br />
[2]  Nathan Stevenson, Karoliina Tapani, Leena Lauronenand Sampsa Vanhatalo, “A dataset of neonatal EEG recordings with seizures annotations”. Zenodo, Jun. 05, 2018. doi: 10.5281/zenodo.2547147.
