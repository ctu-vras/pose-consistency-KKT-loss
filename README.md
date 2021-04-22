# Pose predicting KKT-loss for weakly supervised learning of robot-terrain interaction model
Code for RA-L publication

## Requirements

Code was implemented for python 2.7.
### Required libraries
numpy  
scipy  
Torch  
tensorboardX  

## Dataset

Please download the dataset from https://drive.google.com/drive/folders/1qMwzeyThEgAincA_ldWhTd5rZHmGAJp7

Dataset is expected to be located in "../data/" folder.

## Pretrained models

We provide all pretrained models in folder weights

## Training 

The rigid terrain prediction (experiment 1):

script train_d2rpz.py will train q_omega network  
script train_s2d.py will train h_theta network  
script train_s2d_rpz.py will train h_theta network by pose-predictiong loss backpropagation  (Section 3.2)  
script train_s2d_kkt.py will train h_theta network by kkt loss backpropagation(Section 3.1)

The flexible terrain prediction (experiment 2):

script train_sf2d.py will train h_theta network  
script train_sf2d_rpz.py will train h_theta network by pose-predictiong loss backpropagation  (Section 3.2)  
script train_sf2d_kkt.py will train h_theta network by kkt loss backpropagation(Section 3.1)



## Evaluation

Running the script eval.py will produce the same results as shown in the paper in Table 1  

Running the script eval_flexible.py will produce the same results as shown in the paper in Table 2  
