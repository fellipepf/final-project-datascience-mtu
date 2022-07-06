

<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>Deep learning and machine learning using actigraphy signals for schizophrenia classification</strong>  
  
</h1>  
<h1 align="center">  
Fellipe Paes Ferreira, Aengus Daly  
</h1>  
  
---  
  ## Table of contents
1. [Introduction](#introduction)  
2. [Software requirements](#software-requirements)  
3. [Software build](#software-build)  
4. [Software integration](#software-integration)  

---  
## Introduction
  
This study aims to identify the presence of schizophrenia through the analysis of data collected through a wristwatch with a motor activities sensor using machine learning techniques.  
  
---  
  


  
## Software requirements
Python 3.9  
  
  
## Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/fellipepf/final-project-datascience-mtu.git
 
```  
  
## Project files description [![](./docs/img/pin.svg)](#software-build)  
  
* Feature Engineering  
<table>  
   <tr>   
      <td>File</td>   
      <td>Description</td>  
   </tr>  
  
   <tr>  
      <td>psykose_feature_engineering.py</td>   
      <td>Reads the raw files of control and patients and generate the features</td>  
   </tr>  
</table>  
  
* Machine Learning  
<table>  
   <tr>   
      <td>File</td>   
      <td>Description</td>  
   </tr>  
  
   <tr>  
      <td>

[psykose_machine_learning_models.py](./code/psykose_machine_learning_models.py)
     </td>   
      <td>Contains the ML models</td>  
   </tr>  
   <tr>  
      <td>psykose_machine_learning_hyperparameters.py</td>   
      <td>Perform the models hyper parameters tuning</td>  
   </tr>  
</table>  
  
* Deep Learning  
<table>  
   <tr>   
      <td>File</td>   
      <td>Description</td>  
   </tr>  
  
   <tr>  
      <td>psyche_2_loo_time_cat_cnn2d_6_5_noval_6t.py</td>   
      <td>Contains the Deep Learning model</td>  
   </tr>  
  
</table>

* Utils 
<table>  
   <tr>   
      <td>File</td>   
      <td>Description</td>  
   </tr>  
  
   <tr>  
      <td>my_metrics.py</td>   
      <td>Class with the metrics used in the project</td>  
   </tr>  
</table>  

