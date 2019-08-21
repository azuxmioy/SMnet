Spatiotemporal Multimodal Network for Dynamic Gesture Recognition
====

Author: Hsuan-I Ho (hohs@student.ethz.ch), Chi-Ching Hsu (hsuch@student.ethz.ch)

## Abstract

![](https://i.imgur.com/PiUrVC3.png)

Human gesture recognition is among interesting topics of visual understanding, which benefits a variety of applications like user interface design and robotics perception. With the goal of deriving an effective model to recognize human gestures, we propose a deep neural network architecture for describing and recognizing important spatial-temporal information across different video data domains. Our proposed network is realized by using a technique of transfer learning, in which the network can be trained and evaluated on a relative small dataset while possible overfitting can be also mitigated. In our experiments, our method is able to achieve state-of-the-art accuracy on the challenging gesture recognition benchmark.

## Prerequisites
Our code is implemented using the following modules on Leonhard cluster:
* StdEnv
* gcc/4.8.5
* openblas/0.2.19
* cuda/10.0.130
* cudnn/7.5
* nccl/2.3.7-1
* jpeg/9b
* libpng/1.6.27
* python_gpu/3.7.1
* tmux/2.6

You can simply load this setting using the following commands on Leonhard:

```
module load python_gpu/3.7.1
module load tmux/2.6
```

## How to reproduce results on our report

### Single model prediction

You can load our models and generate the .csv prediction file with commands

```
python3 restore_and_evaluate.py -S models -M 1559977294 -C 82280
```

This would generate a .csv file with an accuracy of 0.8804.

### Assemble prediction

We also saved softmax predictions from many variants of our models in the folder "assemble". To generate the averaged prediction of all softmax labels you can run to code in the folder which would generate a result of 0.9126.

```
python3 assemble.py
```

## How to retrain our network to get a similar result

We have preprocessed the provided data and saved it to our own tfrecord files. In addition, we also applied pretrained weights on our feature extractors. To run our training scripts, you should download the following files and save them into the corresponding folders in "tmp". 

* [Pretrained models (1.8GB)](https://polybox.ethz.ch/index.php/s/Q6aVKEC7QfyhA1w)
* [Preprocessed data (15.5GB)](https://polybox.ethz.ch/index.php/s/w5Oeu5UnKHyMVdE)

The "tmp" folder should have following subdirectories:
```
tmp
 ⊢ C3D/ (contains C3D pretrained weights)
 ⊢ log/ (for saving tensorboard logs and models)
 ⊢ newdata/ (contains preprocessed tfrecords)
```

Note that our model should be trained on a **TeslaV100_SXM2_32GB** or there will be an out of memory issue. We train our model using the interactive session with 40000 MB memories (for the purpose of suffling buffers), and usually it takes about 3~4 4-hour sessions to achieve a satisfactory result. 


To be detailed, we use the following command to request an interactive session
```
bsub -Is -W 4:00 -n 1 -R "rusage[mem=40000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" bash
```
and run the training script (We have already adjusted the parameters in config.py)
```
python3 training.py
```

Usually we decrease the learning rate only in the third and forth training session for finetuning, i.e. change the parameters in config.py
```
config['learning_rate'] = 5e-5 ----> 1e-5
config['pretrain_lr'] = 1e-5   ----> 2e-6
```