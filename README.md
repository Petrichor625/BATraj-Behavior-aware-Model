# BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving

This repository contains the official implementation of  **BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving**, accepted by the Association for the Advancement of Artificial Intelligence (AAAI) 2024.

## Highlights
- Introducing a real-time dynamic geometric graph method for the continuous representation of driving behavior in trajectory prediction for autonomous driving.
- Pioneering a pooling strategy that aligns with human observation instincts by capturing the location of traffic agents using polar coordinates.
- Introducing the MoCAD dataset with its unique right-hand-drive system and mandatory left-hand driving, enriching trajectory prediction studies in complex urban and campus scenarios.
- The proposed model demonstrates exceptional robustness and adaptability in various traffic scenarios, including highways, roundabouts, campuses, and busy urban locales.
  

# ⚠️ **Important Notice: Code Version Update Pending** ⚠️
**🚨 Apologies for the inconvenience!**  
There is an issue with the current version of the code in this repository. Unfortunately, we’re unable to upload the latest corrected version right now.

For those aiming to replicate or build upon our results, we recommend checking out our alternative project:  
👉 **[HLTP Project on GitHub](https://github.com/Petrichor625/HLTP)** 👈

This project includes **improved methods** and **refined code** that may better support your work. 

Please don’t hesitate to reach out if you have any questions—thank you for your understanding and patience!



## To-do List

###### **Note**

- [x] [2023.5.25] Creating the repository for BAT
- [x] [2023.8.05] Open source BAT code
- [x] [2023.11.28] Update Readme
- [x] [2023.12.09] Update project code
- [x] [2024.3.14] Update visualization code (See more details in our new work HLTP)
- [ ] Upload the latest version of the code (**if you want to quickly reproduce better results while waiting, please refer to our latest work HLTP**)



## Install

The model install in Ubuntu 20.04, cuda11.7

**1. Clone this repository**: clone our model use the following code 

```
git clone https://github.com/Petrichor625/BATraj-Behavior-aware-Model.git
cd Behavior_aware file
```

**2. Implementation Environment**: The model is implemented by using Pytorch. We share our anaconda environment in the folder 'environments', then use this command to implement your environment.

```
cd environments
conda env create -f environment.yml
```

If this command cannot implement your conda environment well, try to install again the specific package separately.

```
conda create -n Behaivor_aware python=3.8
conda activate Behavior-aware
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```



## Download and Pre-processing Datasets

Before starting the training process,   you can choose one of the datasets to train the model,  and you need to prepare the dataset and follow the steps below.



**Next Generation Simulation (NGSIM) Dataset**

<u>The esteemed NGSIM dataset is available at the Releases in this repository.</u>

###### **Note**

1. [2023.5.25] Like the baselines, the NGSIM dataset in our work is segmented in the same way as in the most widely used work [Deo and Trivedi, 2018], so that comparisons can be made.

2. [2023.5.25] The maneuver-based test set is only a true subset of the overall test set. We select well-defined maneuvers from the overall test set and divide them into the maneuver-based test set, omitting some unknown maneuvers such as zigzag driving, tentative driving, etc.



## Train 

In this section, we will explain how to train the model.

Please keep your model in a file named `models` and change your hyperparameter in models.py

###### Note

You can view or change the network parameters from the `model_args.py` file

##### Begin to Train

```
python3 train.py
```

The training logs will keep in the file name `Train_logs`, you can use tensorboard to check your training process.

```
tensorboard --logdir /YourPath/Train_log/yourmodel
```

**Save trained model**

In addition, the trained model will be stored in the path you set, which can be set from the `model_fname` in the `train.py` file 

```
model_fname = 'path/to/save/model'
```






## Qualitative results

We are preparing a script for generating these visualizations:

 ````
 Code for qualitative results coming soon
 ````
 ![image](https://github.com/Petrichor625/BATraj-Behavior-aware-Model/blob/main/Figures/Qualitative%20results.gif)


## Main contributions

This work aims to introduce a novel **lightweight and map-free model** that eliminates the need for labor-intensive high-definition (HD) maps and costly manual annotation by using only historical trajectory data in the polar coordinate system and a DGG-based method to capture continuous driving behavior. It can be summarized as follows:

1. We present a novel dynamic geometric graph approach that eliminates the need for manual labeling during training. This method addresses the challenges of labeling non-continuous behaviors and selecting appropriate time windows, while effectively capturing continuous driving behavior. Inspired by traffic psychology, decision theory, and driving dynamics, our model incorporates centrality metrics and behavior-aware criteria to provide enhanced flexibility and accuracy in representing driving behavior. To the best of our knowledge, this is the first attempt to incorporate **continuous representation** of behavioral knowledge} in trajectory prediction for AVs.
2. We propose a novel pooling mechanism, aligned with human observational instincts, that extracts vehicle positions in **polar coordinates**. It simplifies the representation of direction and distance in Cartesian coordinates, accounts for road curvature, and allows modeling in complex scenarios such as roundabouts and intersections.
3. We introduce a new Macao Connected Autonomous Driving (MoCAD) dataset, sourced from a L5 autonomous bus with over 300 hours across campus and busy urban routes. Characterized by its unique **right-hand-drive system**, MoCAD, set to be publicly available, is pivotal for research in right-hand-drive dynamics and enhancing trajectory prediction models.
4. Our model significantly outperforms the SOTA baseline models when tested on the NGSIM, HighD, RounD, and MoCAD datasets, respectively. Remarkably, it maintains impressive performance even when trained on only **25.0% of the dataset**, demonstrating exceptional robustness and adaptability in various traffic scenarios, including **highways**, **roundabouts**, and **busy urban locales**.



## Citation
**BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving**, proceeding on the Association for the Advancement of Artificial Intelligence (AAAI) 2024.

```
@inproceedings{liao2024bat,
  title={Bat: Behavior-aware human-like trajectory prediction for autonomous driving},
  author={Liao, Haicheng and Li, Zhenning and Shen, Huanming and Zeng, Wenxuan and Liao, Dongping and Li, Guofa and Xu, Chengzhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={9},
  pages={10332--10340},
  year={2024}
}
```
 


