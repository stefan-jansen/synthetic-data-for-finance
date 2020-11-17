# Generative Adversarial Nets for Synthetic Time Series Data

This repo shows how to create synthetic time-series data using generative adversarial networks (GAN). GANs train a generator and a discriminator network in a competitive setting so that the generator learns to produce samples that the discriminator cannot distinguish from a given class of training data. The goal is to yield a generative model capable of producing synthetic samples representative of this class.
While most popular with image data, GANs have also been used to generate synthetic time-series data in the medical domain. Subsequent experiments with financial data explored whether GANs can produce alternative price trajectories useful for ML training or strategy backtests. 

We replicate the 2019 NeurIPS [Time-Series GAN](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) paper by Jinsung Yoon, et al., to illustrate the approach and demonstrate the results. The material is based on the 2<sup>nd</sup> edition of my book on [Machine Learning for Trading]((https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d)) (see [GitHub repo](https://github.com/stefan-jansen/machine-learning-for-trading)).  

<p align="center">
<img src="https://i.imgur.com/W1Rp89K.png" width="60%">
</p>

## Content

1. [Generative adversarial networks for synthetic data](#generative-adversarial-networks-for-synthetic-data)
    * [Comparing generative and discriminative models](#comparing-generative-and-discriminative-models)
    * [Adversarial training: a zero-sum game of trickery](#adversarial-training-a-zero-sum-game-of-trickery)
2. [Code example: TimeGAN: Adversarial Training for Synthetic Financial Data](#code-example-timegan-adversarial-training-for-synthetic-financial-data)
    * [Learning the data generation process across features and time](#learning-the-data-generation-process-across-features-and-time)
    * [Combining adversarial and supervised training with time-series embedding](#combining-adversarial-and-supervised-training-with-time-series-embedding)
    * [The four components of the TimeGAN architecture](#the-four-components-of-the-timegan-architecture)
    * [Implementing TimeGAN using TensorFlow 2](#implementing-timegan-using-tensorflow-2)
    * [Evaluating the quality of synthetic time-series data](#evaluating-the-quality-of-synthetic-time-series-data)
3. [Resources](#resources)
    * [How GAN's work](#how-gans-work)
    * [Implementation](#implementation)
    * [The rapid evolution of the GAN architecture zoo](#the-rapid-evolution-of-the-gan-architecture-zoo)
    * [Applications](#applications)

## Generative adversarial networks for synthetic data

The [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d) mostly focuses on supervised learning algorithms that receive input data and predict an outcome, which we can compare to the ground truth to evaluate their performance. Such algorithms are also called discriminative models because they learn to differentiate between different output values.
Generative adversarial networks (GANs) are an instance of generative models like the variational autoencoder covered in [Chapter 20](https://github.com/stefan-jansen/machine-learning-for-trading/tree/master/20_autoencoders_for_conditional_risk_factors).

### Comparing generative and discriminative models

Discriminative models learn how to differentiate among outcomes y, given input data X. In other words, they learn the probability of the outcome given the data: p(y | X). Generative models, on the other hand, learn the joint distribution of inputs and outcome p(y, X). 

While generative models can be used as discriminative models using Bayes Rule to compute which class is most likely (see [Chapter 10](https://github.com/stefan-jansen/machine-learning-for-trading/tree/master/10_bayesian_machine_learning)), it appears often preferable to solve the prediction problem directly rather than by solving the more general generative challenge first.

### Adversarial training: a zero-sum game of trickery

The key innovation of GANs is a new way of learning the data-generating probability distribution. The algorithm sets up a competitive, or adversarial game between two neural networks called the generator and the discriminator.

<p align="center">
<img src="https://i.imgur.com/0vuUsY0.png" width="80%">
</p>

## Code example: How to build a GAN using TensorFlow 2

To illustrate the implementation of a generative adversarial network using Python, we use the deep convolutional GAN (DCGAN) example discussed earlier in this section to synthesize images from the fashion MNIST dataset that we first encountered in Chapter 13. 

The notebook [deep_convolutional_generative_adversarial_network](https://github.com/stefan-jansen/machine-learning-for-trading/blob/master/21_gans_for_synthetic_time_series/01_deep_convolutional_generative_adversarial_network.ipynb) illustrates the implementation of a GAN using Python. It uses the Deep Convolutional GAN (DCGAN) example to synthesize images from the fashion MNIST dataset

## Code example: TimeGAN: Adversarial Training for Synthetic Financial Data

Generating synthetic time-series data poses specific challenges above and beyond those encountered when designing GANs for images. 
In addition to the distribution over variables at any given point, such as pixel values or the prices of numerous stocks, a generative model for time-series data should also learn the temporal dynamics that shapes how one sequence of observations follows another (see also discussion in Chapter 9: [Time Series Models for Volatility Forecasts and Statistical Arbitrage](../09_time_series_models)).

Very recent and promising research by Yoon, Jarrett, and van der Schaar, presented at NeurIPS in December 2019, introduces a novel [Time-Series Generative Adversarial Network](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf) (TimeGAN) framework that aims to account for temporal correlations by combining supervised and unsupervised training. 
The model learns a time-series embedding space while optimizing both supervised and adversarial objectives that encourage it to adhere to the dynamics observed while sampling from historical data during training. 
The authors test the model on various time series, including historical stock prices, and find that the quality of the synthetic data significantly outperforms that of available alternatives.

### Learning the data generation process across features and time

A successful generative model for time-series data needs to capture both the cross-sectional distribution of features at each point in time and the longitudinal relationships among these features over time. 
Expressed in the image context we just discussed, the model needs to learn not only what a realistic image looks like, but also how one image evolves from the next as in a video.

### Combining adversarial and supervised training with time-series embedding

Prior attempts at generating time-series data like the recurrent (conditional) GAN relied on recurrent neural networks (RNN, see Chapter 19, [RNN for Multivariate Time Series and Sentiment Analysis](../19_recurrent_neural_nets)) in the roles of generator and discriminator. 

TimeGAN explicitly incorporates the autoregressive nature of time series by combining the unsupervised adversarial loss on both real and synthetic sequences familiar from the DCGAN example with a stepwise supervised loss with respect to the original data. 
The goal is to reward the model for learning the distribution over transitions from one point in time to the next present in the historical data.

### The four components of the TimeGAN architecture

The TimeGAN architecture combines an adversarial network with an autoencoder and has thus four network components as depicted in Figure 21.4:
Autoencoder: embedding and recovery networks
Adversarial Network: sequence generator and sequence discriminator components
<p align="center">
<img src="https://i.imgur.com/WqoXbr8.png" width="80%">
</p>

### Implementing TimeGAN using TensorFlow 2

In this section, we implement the TimeGAN architecture just described. The authors provide sample code using TensorFlow 1 that we port to TensorFlow 2. Building and training TimeGAN requires several steps:
1. Selecting and preparing real and random time series inputs
2. Creating the key TimeGAN model components
3. Defining the various loss functions and train steps used during the three training phases
4. Running the training loops and logging the results
5. Generating synthetic time series and evaluating the results

The notebook [TimeGAN_TF2](02_TimeGAN_TF2.ipynb) shows how to implement these steps.

### Installation

Using a GPU is recommended to speed up training. There are several options to run the notebook:
1) Use a [Docker](https://docs.docker.com/get-started/overview/) image provided by TensorFlow with either CPU or GPU support. See [instructions](https://www.tensorflow.org/install/docker). 
    - To start the container configured with TensorFlow and mount the project directory in the `/home` directory, run the following command in this repo's root folder on your machine:
        - With GPU support (using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) as describe in the linked [instructions](https://www.tensorflow.org/install/docker)):
            ```bash
            docker run --gpus all -it -v $(pwd):/home -p 8888:8888 --name ml4t tensorflow/tensorflow:latest-gpu-jupyter bash
          ```
      - With CPU support:
          ```bash
          docker run -it -v $(pwd):/home -p 8888:8888 --name ml4t tensorflow/tensorflow:latest-gpu-jupyter bash
          ```
    - Change into the `/home` folder of your container using `cd /home`
    - Run the install script to get some requisite packages: `./install.sh`
    - Then, launch the jupyter server to work with the notebooks as usual:
        ```bash
        jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
        ```
2) Create a virtual environment using the `requirements.txt` file (Ubuntu only; other OS requires modifying the content).

### Evaluating the quality of synthetic time-series data

The TimeGAN authors assess the quality of the generated data with respect to three practical criteria:
1. **Diversity**: the distribution of the synthetic samples should roughly match that of the real data
2. **Fidelity**: the sample series should be indistinguishable from the real data, and 
3. **Usefulness**: the synthetic data should be as useful as their real counterparts for solving a predictive task

The authors apply three methods to evaluate whether the synthetic data actually exhibits these characteristics:
1. **Visualization**: for a qualitative diversity assessment of diversity, we use dimensionality reduction (principal components analysis (PCA) and t-SNE, see Chapter 13) to visually inspect how closely the distribution of the synthetic samples resembles that of the original data
2. **Discriminative Score**: for a quantitative assessment of fidelity, the test error of a time-series classifier such as a 2-layer LSTM (see Chapter 18) let’s us evaluate whether real and synthetic time series can be differentiated or are, in fact, indistinguishable.
3. **Predictive Score**: for a quantitative measure of usefulness, we can compare the test errors of a sequence prediction model trained on, alternatively, real or synthetic data to predict the next time step for the real data.

The notebook [evaluating_synthetic_data](03_evaluating_synthetic_data.ipynb) contains the relevant code samples.

## Resources

### How GAN's work

- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf), Ian Goodfellow, 2017
- [Why is unsupervised learning important?](https://www.quora.com/Why-is-unsupervised-learning-important), Yoshua Bengio on Quora, 2018
- [GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation](https://www.groundai.com/project/gan-lab-understanding-complex-deep-generative-models-using-interactive-visual-experimentation/), Minsuk Kahng, Nikhil Thorat, Duen Horng (Polo) Chau, Fernanda B. Viégas, and Martin Wattenberg, IEEE Transactions on Visualization and Computer Graphics, 25(1) (VAST 2018), Jan. 2019
    - [GitHub](https://poloclub.github.io/ganlab/)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Ian Goodfellow, et al, 2014
- [Generative Adversarial Networks: an Overview](https://arxiv.org/pdf/1710.07035.pdf), Antonia Creswell, et al, 2017
- [Generative Models](https://blog.openai.com/generative-models/), OpenAI Blog

### Implementation

- [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
- [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan)
- [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN), numerous Keras GAN implementations
- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN), numerous PyTorch GAN implementations


### The rapid evolution of the GAN architecture zoo

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf), Luke Metz et al, 2016
- [Conditional Generative Adversarial Net](https://arxiv.org/pdf/1411.1784.pdf), Medhi Mirza and Simon Osindero, 2014
- [Infogan: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf), Xi Chen et al, 2016
- [Stackgan: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242.pdf), Shaoting Zhang et al, 2016
- [Photo-realistic Single Image Super-resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf), Alejando Acosta et al, 2016
- [Unpaired Image-to-image Translation Using Cycle-consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf), Juan-Yan Zhu et al, 2018
- [Learning What and Where to Draw](https://arxiv.org/abs/1610.02454), Scott Reed, et al 2016
- [Fantastic GANs and where to find them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them)

### Applications

- [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633), Cristóbal Esteban, Stephanie L. Hyland, Gunnar Rätsch, 2016
    - [GitHub Repo](https://github.com/ratschlab/RGAN)
- [MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks](https://arxiv.org/pdf/1901.04997.pdf), Dan Li, Dacheng Chen, Jonathan Goh, and See-Kiong Ng, 2019
    - [GitHub Repo](https://github.com/LiDan456/MAD-GANs)
- [GAN — Some cool applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900), Jonathan Hui, 2018
- [gans-awesome-applications](https://github.com/nashory/gans-awesome-applications), curated list of awesome GAN applications



