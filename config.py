"""Copyright (c) 2019 AIT Lab, ETH Zurich, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import time
import os

config = dict()

##################################################################
# Please note that the following fields will be set by our scripts to re-train and re-evaluate your model.

# Where experiment results are stored.
config['log_dir'] = 'tmp/log'

# In case your pre/post-processing scripts generate intermediate results, you may use config['tmp_dir'] to store them.
config['tmp_dir'] = 'tmp/'

# Path to training, validation and test data folders.
config['data_dir'] = 'tmp/newdata'
##################################################################
# You can modify the rest or add new fields as you need.
config['finetune'] = False
config['pretrained'] = None

# Dataset statistics. You don't need to change unless you use different splits.
config['num_test_samples'] = 2174
config['num_validation_samples'] = 1765
config['num_training_samples'] = 5722

# Hyper-parameters and training configuration.
config['batch_size'] = 4
config['learning_rate'] = 5e-5
config['pretrain_lr'] = 1e-5

# Learning rate is annealed exponentially in 'exponential' case. You can change annealing configuration in the code.
config['learning_rate_type'] = 'fixed'  # 'fixed' or 'exponential'

#config['num_steps_per_epoch'] = int(config['num_training_samples']/config['batch_size'])
config['num_steps_per_epoch'] = int( (config['num_training_samples'] + config['num_validation_samples'] )/config['batch_size'])


config['normalize_data'] = True
config['num_epochs'] = 30
config['evaluate_every_step'] = config['num_steps_per_epoch']
config['checkpoint_every_step'] = config['num_steps_per_epoch']
config['print_every_step'] = 50

# Here I provide three common techniques to calculate sequence loss.
# (1) 'last_logit': calculate loss by using only the last step prediction.
# (2) 'average_logit': calculate loss by using average of predictions across all steps.
# (3) 'average_loss': calculate loss for each time-step by using the same sequence label.
config['loss_type'] = 'average_loss'  # 'last_logit', 'average_logit', 'average_loss'.

# Dataset and Input Pipeline Configuration
config['inputs'] = {}
config['inputs']['num_epochs'] = config['num_epochs']
config['inputs']['batch_size'] = config['batch_size']

config['feature_len'] = 16
config['subsample'] = 1

# CNN model parameters
config['cnn'] = {}
config['cnn']['num_filters'] = [16, 32, 64, 128, 256]  # Number of filters for every convolutional layer.
config['cnn']['filter_size'] = [3, 3, 3, 3, 3]  # Kernel size. Assuming kxk kernels.
config['cnn']['num_hidden_units'] = 256  # Number of units in the last dense layer, i.e. representation size.
config['cnn']['dropout_rate'] = 0
config['cnn']['num_class_labels'] = 20
config['cnn']['batch_size'] = config['batch_size']
config['cnn']['loss_type'] = config['loss_type']
config['cnn']['subsample'] = config['subsample']

# RNN model parameters
config['rnn'] = {}
config['rnn']['num_hidden_units'] = 256  # Number of units in an LSTM cell.
config['rnn']['dropout_rate'] = 0.3
config['rnn']['num_layers'] = 2  # Number of LSTM stack.
config['rnn']['num_class_labels'] = 20
config['rnn']['batch_size'] = config['batch_size']
config['rnn']['loss_type'] = config['loss_type']
# c3d len
config['rnn']['feature_len'] = config['feature_len']
config['rnn']['subsample'] = config['subsample']

# C3D model parameters
config['c3d'] = {}
config['c3d']['feature_len'] = config['feature_len'] 
config['c3d']['rgb_num_hidden_units'] = 256  
config['c3d']['depth_num_hidden_units'] = 128  
config['c3d']['sk_num_hidden_units'] = 128  
config['c3d']['pretrain_rgb'] = 'tmp/C3D/c3d_rgb.chkp'
config['c3d']['pretrain_depth'] = 'tmp/C3D/c3d_depth.chkp'
config['c3d']['pretrain_sk'] = 'tmp/C3D/c3d_sk.chkp'

config['c3d']['dropout_rate'] = 0.3
config['c3d']['num_class_labels'] = 20
config['c3d']['loss_type'] = config['loss_type']
config['c3d']['subsample'] = config['subsample']


# R3D model parameters
config['r3d'] = {}
config['r3d']['feature_len'] = config['feature_len'] 
config['r3d']['num_hidden_units'] = 256  # Number of units in the last dense layer, i.e. representation size.
config['r3d']['dropout_rate'] = 0
config['r3d']['num_class_labels'] = 20
config['r3d']['loss_type'] = config['loss_type']

# Skeleton model parameters
config['sk_encoder'] = {}
config['sk_encoder']['feature_len'] = config['feature_len'] 
config['sk_encoder']['num_hidden_units'] = 256  # Number of units in the last dense layer, i.e. representation size.
config['sk_encoder']['dropout_rate'] = 0
config['sk_encoder']['num_class_labels'] = 20
config['sk_encoder']['loss_type'] = config['loss_type']

# You can set descriptive experiment names or simply set empty string ''.
config['model_name'] = 'skeleton_code'


# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
model_folder_name = timestamp if config['model_name'] == '' else timestamp + "_" + config['model_name']
config['model_id'] = timestamp
config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
print("Writing to {}\n".format(config['model_dir']))
config['checkpoint_id'] = None
