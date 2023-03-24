# BATraj-Behavior-aware-Model
This repository is the code of BATraj: A Behavior-aware Trajectory Prediction Model for Autonomous Driving.

**The NGSIM dataset is available at the _Releases_ in this repository.**

## Introduction

This project introduce a behavior-aware model for trajectory prediction that incorporates theories and findings from the fields of human behavior, decision-making, theory of mind, etc. This model is comprised of behavior-aware, interaction-aware, priority-aware, and position-aware modules that analyze and interpret a variety of inputs, perceive and comprehend underlying interactions, and take into account uncertainty and variability in prediction. We evaluate the performance of our model using the NGSIM dataset and show that it outperforms current state-of-the-art baselines in terms of prediction accuracy and efficiency. Even when trained on a reduced portion of the training data, specifically 25%, our model outperforms all baselines, demonstrating its robustness and efficiency in predicting future vehicle trajectories and the potential to lower the amount of data required for training autonomous vehicles, particularly in corner cases.



## Our model

This model comprises of four innovative modules - a behavior-aware module, an interaction-aware module, a priority-aware module, and a position-aware module - each designed to enhance the sophistication and nuance of the model's understanding of driver behavior and vehicle interactions on the road.

**The behavior-aware module**, in particular, is a key component of this model. Instead of resorting to a simplistic classification of behaviors into two or three distinct typologies, it utilizes a continuous representation of behavioral information, rooted in dynamic geometric graph theory, to offer unparalleled flexibility and scalability in dynamic driving contexts. This allows autonomous vehicles to anticipate and respond to the actions of other drivers in a more intricate and elaborate manner. 

**The interaction-aware module**, on the other hand, takes into account the interactions between the AV and other vehicles in the environment, utilizing Long Short-Term Memory Networks (LSTMs) encoder to process historical track information for the ego vehicle and surrounding vehicles, thus enabling the ego vehicle to have a better understanding of the potential interactions with other vehicles. 

**The priority-aware module** evaluates the importance of different vehicles and allows the ego vehicle to prioritize its attention and response to certain vehicles over others, based on their relevance to the ego vehicle's trajectory. 

**The position-aware module** encodes and learns the dynamic location of the ego vehicle, providing additional context for the prediction of the ego vehicle's trajectory. Furthermore, we introduce a Polar coordinate system to accommodate the relative distance among various vehicles and scenes, which provides a flexible way to adapt to heterogeneous input data.



![image](https://github.com/Petrichor625/BATran-Behavior-aware-Model/blob/main/framework.png)



## Baseline

**[S-LSTM](https://ieeexplore.ieee.org/document/7780479)**: This method integrates a social pooling mechanism to the LSTM model.

**[S-GAN:](https://ieeexplore.ieee.org/document/8578338)** This baseline presents a pooling strategy based on generative adversarial networks (GAN).

**[CS-LSTM](https://ieeexplore.ieee.org/document/8575356):** This approach uses a convolutional layer for the pooling mechanism with a maneuver classifier provided.

**[MATF-GAN](https://ieeexplore.ieee.org/document/8953520):** This model employs convolutional fusion to capture the spatial interaction between agents and scene context, and utilizes GAN to generate predicted trajectories. 

**[NLS-LSTM](https://ieeexplore.ieee.org/document/8813829):** The non-local social pooling scheme captures the social interaction by combining both local and non-local operations to produce context vectors for social pooling.

**[PiP](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_36):** The planning-informed prediction model integrates trajectory prediction with target vehicle planning to predict the trajectory by using candidate target vehicle trajectories as a fundamental component.

**[MHA-LSTM](https://ieeexplore.ieee.org/document/9084255):** The multi-head attention social pooling scheme uses the multi-head dot product attention mechanism to predict the trajectory

**[TS-GAN](https://ieeexplore.ieee.org/document/9151374):** This approach uses a social convolution mechanism and recurrent social module for vehicle trajectory prediction.

**[STDAN](https://ieeexplore.ieee.org/document/9767719):** The model employs a spatial-temporal dynamic attention network to predict trajectory prediction.

## 


## Conclusion

Predicting the trajectories of surrounding vehicles with a high degree of accuracy is a fundamental challenge that must be addressed in the quest for fully autonomous vehicles. To address this challenge, we present a behavior-aware model for trajectory prediction that leverages a wide range of theories and findings from fields such as human behavior, decision-making, and theory of mind. Our model is built on a modular architecture comprising four key components: the behavior-aware module, the interaction-aware module, the priority-aware module, and the position-aware module. These modules work in tandem to analyze and interpret various inputs, perceive and comprehend underlying interactions, and take into account uncertainty and variability in prediction. The behavior-aware module uses dynamic graphs to model and predict human driving behaviors. The interaction-aware module employs stacks of LSTMs to model the interactions and dependencies between vehicles. The priority-aware module utilizes the attention mechanism to model the likelihood of different vehicles being given priority in various situations. Finally, the position-aware module leverages spatial-temporal techniques to model and encode the positions of vehicles. 

We evaluate the performance of our model using the NGSIM dataset, a widely-used dataset for evaluating the performance of trajectory prediction models. **Our results demonstrate that our model outperforms current state-of-the-art baselines in terms of prediction accuracy and efficiency, and is able to achieve this performance even when trained on a reduced portion of the training data (specifically, 25%).** This highlights the robustness and efficiency of our model in predicting future vehicle trajectories, and suggests that it has the potential to significantly reduce the amount of data required for training autonomous vehicles, particularly in challenging or unusual situations.

In conclusion, the behavior-aware model represents a major advancement in the development of autonomous vehicles capable of predicting trajectories with the same level of proficiency as human drivers. In the future, there are several areas of research that we plan to explore in order to further improve the performance of our model. These include incorporating additional sources of data and information, such as sensor data from on-board cameras and lidar, as well as exploring new machine learning techniques and algorithms that may improve the accuracy and efficiency of our model. We also plan to investigate how our model can be integrated into larger autonomous vehicle systems, and how it can be used to improve the safety and reliability of autonomous vehicles in real-world scenarios.

### An overview of the Polar coordinates system and the proposed behavior-aware and position-aware modules.###
![image](https://github.com/Petrichor625/BATran-Behavior-aware-Model/blob/main/polar.png)

### Multi-modal maneuver prediction framework with corresponding probability outputs.###
![image](https://github.com/Petrichor625/BATran-Behavior-aware-Model/blob/main/trajectory-3.png)
