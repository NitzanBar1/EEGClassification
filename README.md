# EEGClassification

Image Classification of EEG signals using PyTorch

Video:

[YouTube](https://youtu.be/V5hxXmG1A9U) - https://youtu.be/V5hxXmG1A9U


- [EEGClassification](#EEGClassification)
  * [Prerequisites](#prerequisites)
  * [Files in The Repository](#files-in-the-repository)
  * [Introduction](#introduction)
  * [Dataset Acquisition](#dataset-acquisition) 
  * [Proposed CNN Model](#proposed-cnn-model)
  * [Results](#results)
  * [References](#references)


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`|  `0.4.1`|


## Files in the repository
|File name         | Purpsoe |
|----------------------|------|
|`EEGClassification.ipynb`| main application for training the model|
|`utils.py`| utils functions for loading and pre-processing the data|


## Introduction
Vision is one of the most significant parts in the human perception framework. When the eyes get visual incitement, neural spikes are delivered to the brain. The interpretation of these neural spikes is becoming an interesting research work in the era of computer vision. Paradigms leveraging stimuli evoked EEG signal have been used to investigate and analyse the complexity of EEG signal for object detection and for classification tasks using machine learning methods. 
Deep learning-based models have outperformed conventional methodologies that eliminate the manual feature extraction step. Convolutional neural networks (CNNs) specifically have become a famous deep learning-based approach for learning discriminative features for classification tasks. Past experiments at applying CNNs to stimuli evoked EEG signal classification have utilized domain specific feature representations to decrease the required information for classification. 
This report aims at providing a CNN-based framework for classification of visually evoked stimuli. Figure 1 describes the overall workflow of the proposed architecture. 
As a very initial step, we have explored the utilization of CNNs for multi-classification of EEG signals recorded while a subject is viewing image of digits from 0 to 9 as stimuli. In the second step, the acquired EEG signals are then pre-processed using basic filtering process in order to remove artifacts. The EEG signals are then used as input to the proposed CNN model for a 10-class classification task representing the 10 different digits (0–9). 

![intro](https://github.com/NitzanShitrit/EEGClassification/blob/master/images/intro.PNG)


## Dataset Acquisition
MindBigData (The “MNIST” of Brain Digits) is an open database containing 1,207,293 brain signals of 2s each, captured with the stimulus of seeing a digit (from 0 to 9) and thinking about it. The raw EEG signals were captured 128 Hz sampling rate, so there are approximately 256 (128 × 2 secs) data points for each stimulus image of digit 0–9 [1]. In this project, 6500 trails for each digit image, with each trail containing 256 samples for 14 electrode positions, have been used. Overall, 65,000 samples been used, 75% were utilized for training, 15% for validation and 10% was utilized for testing.
Brain locations:
Each EEG device capture the signals via different sensors, located in these areas of the brain.
We used EPOC with 14 channels, as described in the following figure, marked with blue colour.

![brain](https://github.com/NitzanShitrit/EEGClassification/blob/master/images/brain.PNG)
![signals](https://github.com/NitzanShitrit/EEGClassification/blob/master/images/signals.PNG)


## Proposed CNN Model
The network is composed of five convolution blocks and fully connected layers. Each convolution block consists of a convolution layer, a batch normalization, and an exponential linear unit, as shown in the following figure. 
An illustration of the proposed network is shown below:

![cnn](https://github.com/NitzanShitrit/EEGClassification/blob/master/images/cnn.PNG)

C1 and C2 blocks were designed to extract the spectral representation of the EEG input, as it performs convolution across the time dimension, capturing features from each EEG channel independently from the others.
C3 block was designed for performing spatial filtering, as it performs convolutions across the channel dimension. The objective of this layer is to learn the weights of all channels at each time sample.
C4 and C5 blocks are capturing the temporal patterns in each extracted feature maps.  
Dropout has been used in deep neural network training as a regularisation technique to reduce the network tendency to overfit during the training process. 
Therefore, we used it after convolution blocks C3 and C5, with the dropout set to 0.5.


## Results
The proposed CNN model was compiled in Google COLAB using PyTorch packages with 200 epochs. 
The acquired average accuracy through proposed CNN architecture for stimuli evoked EEG signals of classification MindBigData dataset is 21.5%. 
The loss graph and confusion matrix of the proposed CNN model is shown below.
The following table describes the comparison between some previous stimuli evoked EEG classification techniques along with their accuracy results and the associated stimuli that were used for acquiring the EEG signals. 

![graphs](https://github.com/NitzanShitrit/EEGClassification/blob/master/images/graphs.PNG)
![table](https://github.com/NitzanShitrit/EEGClassification/blob/master/images/table.PNG)


## References
*	Mindbigdata dataset. http://www.mindbigdata.com/ (2018)
*	Henry, J. Craig. "Electroencephalography: basic principles, clinical applications, and related fields." Neurology 67.11 (2006): 2092-2092. 
*	Bird, Jordan J., et al. "A deep evolutionary approach to bioinspired classifier optimisation for brain-machine interaction." Complexity 2019 (2019).
*	Jolly, Baani Leen Kaur, et al. "Universal EEG encoder for learning diverse intelligent tasks." 2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM). IEEE, 2019. 
*	Khok, Hong Jing, Victor Teck Chang Koh, and Cuntai Guan. "Deep Multi-Task Learning for SSVEP Detection and Visual Response Mapping." 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2020.
*	Bozal Chaves, Alberto. Personalized image classification from EEG signals using Deep Learning. BS thesis. Universitat Politècnica de Catalunya, 2017.
*	Kwon, Yea-Hoon, Sae-Byuk Shin, and Shin-Dug Kim. "Electroencephalography based fusion two-dimensional (2D)-convolution neural networks (CNN) model for emotion recognition system." Sensors 18.5 (2018): 1383.
*	Ha, Kwon-Woo, and Jin-Woo Jeong. "Motor imagery EEG classification using capsule networks." Sensors 19.13 (2019): 2854.
*	Aznan, Nik Khadijah Nik, et al. "Simulating brain signals: Creating synthetic eeg data via neural-based generative models for improved ssvep classification." 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019.
*	Manor, Ran, and Amir B. Geva. "Convolutional neural network for multi-category rapid serial visual presentation BCI." Frontiers in computational neuroscience 9 (2015): 146.
