"""Copyright (c) 2019 AIT Lab, ETH Zurich, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.contrib import rnn



class Model:
    """
    Base class for sequence models.
    """
    def __init__(self, config, placeholders, mode):
        """
        :param config: dictionary of hyper-parameters.
        :param placeholders: dictionary of input placeholders so that you can pass different modalities.
        :param mode: running mode.
        """

        self.config = config
        self.input_placeholders = placeholders

        assert mode in ["training", "validation", "test"]
        self.mode = mode
        self.is_training = self.mode == "training"
        self.reuse = self.mode == "validation"

        self.input_seq_len = placeholders['length']
        if self.mode is not "test":
            self.input_target_labels = placeholders['label']

        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)

        # Total number of trainable parameters.
        self.num_parameters = 0

        # Batch size and sequence length that are set dynamically.
        self.batch_size_op = None
        self.seq_len_op = None

        # Training objective.
        self.loss = None
        # Logits.
        self.logits = None
        # Label prediction, i.e., what we need to make the submission.
        self.predictions = None
        # Number of the correct predictions.
        self.num_correct_predictions = None
        # Accuracy of the batch.
        self.batch_accuracy = None

        # Set by build_graph method.
        self.input_layer = None
        # This member variable is assumed to be set by build_network() method. It is final layer.
        self.model_output_raw = None

        # These member variables are assumed to be set by build_graph() method.
        # Model outputs with shape [batch_size, seq_len, representation_size]
        self.model_output = None
        # Model outputs with shape [batch_size*seq_len, representation_size]
        self.model_output_flat = None

        self.initializer = tf.glorot_normal_initializer()

    def build_graph(self, input_layer=None):
        """
        Called externally. Builds tensorflow graph by calling build_network. Applies preprocessing on the inputs and
        postprocessing on model outputs.

        :param input_layer: External input. Provides an interface for stacking arbitrary models. For example, RNN model
                            can be fed with output representation of a CNN model.
        """
        raise NotImplementedError('subclasses must override build_graph()')

    def build_network(self):
        """
        Builds internal dynamics of the model. Sets
        """
        raise NotImplementedError('subclasses must override build_network()')

    def build_loss(self):
        """
        Calculates classification loss depending on loss type. We are trying to assign a class label to input
        sequences (i.e., many to one mapping). We need to reduce sequence information into a single step by either
        selecting the last step or taking average over all steps. You are welcome to implement a more sophisticated
        approach.
        """
        
        # Calculate logits
        with tf.variable_scope('logits', reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs_logits:
            dropout_layer = tf.layers.dropout(inputs=self.model_output_flat, rate=self.config['dropout_rate'], training=self.is_training)
            logits_non_temporal = tf.layers.dense(inputs=dropout_layer, units=self.config['num_class_labels'])
            self.logits = tf.reshape(logits_non_temporal, [self.batch_size_op, -1, self.config['num_class_labels']])

            self.variable_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs_logits.name)


        with tf.variable_scope('logits_prediction', reuse=self.reuse, initializer=self.initializer, regularizer=None):
            # Select the last step. Note that we have variable-length and padded sequences.
            if self.config['loss_type'] == 'last_logit':
                self.logits = tf.gather_nd(self.logits, tf.stack([tf.range(self.batch_size_op), tf.cast(self.input_seq_len - 1, tf.int32)], axis=1))
                accuracy_logit = self.logits
            # Take average of time steps.
            elif self.config['loss_type'] == 'average_logit':
                self.logits = tf.reduce_mean(self.logits*self.seq_loss_mask, axis=1)
                accuracy_logit = self.logits
            elif self.config['loss_type'] == 'average_loss':
                accuracy_logit = tf.reduce_mean(self.logits*self.seq_loss_mask, axis=1)
            else:
                raise Exception("Invalid loss type")

        if self.mode is not "test":
            with tf.name_scope("cross_entropy_loss"):

                #self.logits = tf.Print (self.logits, [tf.shape(self.logits)], "Logits: ", summarize = 20)
                #self.logits = tf.Print (self.logits, [tf.shape(self.input_target_labels)], "Label: ", summarize = 20)

                if self.config['loss_type'] == 'average_loss':
                    labels_all_steps = tf.tile(tf.expand_dims(self.input_target_labels, dim=1), tf.reshape([1, tf.reduce_max(self.input_seq_len)], [2]))

                    self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                                 targets=labels_all_steps,
                                                                 weights=self.seq_loss_mask[:, :, 0],
                                                                 average_across_timesteps=True,
                                                                 average_across_batch=True)
                else:
                    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_target_labels))

            with tf.name_scope("accuracy_stats"):
                # Return a bool tensor with shape [batch_size] that is true for the correct predictions.
                correct_predictions = tf.equal(tf.argmax(accuracy_logit, 1), self.input_target_labels)
                # Number of correct predictions in order to calculate average accuracy afterwards.
                self.num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))
                # Calculate the accuracy per mini-batch.
                self.batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Accuracy calculation.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            self.predictions = tf.argmax(accuracy_logit, 1, name="predictions")
            self.softlabel = tf.nn.softmax(accuracy_logit, 1, name="soft_labels")
            
    def get_num_parameters(self):
        """
        :return: total number of trainable parameters.
        """
        # Iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # multiplying dimension values
            self.num_parameters += local_parameters

        return self.num_parameters




class CNNModel(Model):
    """
    Convolutional neural network for sequence modeling.
    - Accepts inputs of rank 5 where a mini-batch has shape of [batch_size, seq_len, height, width, num_channels].
    - Ignores temporal dependency.
    """
    def __init__(self, config, placeholders, mode):
        super(CNNModel, self).__init__(config, placeholders, mode)

        self.input_rgb = placeholders['rgb']

    def build_network(self):
        """
        Stacks convolutional layers where each layer consists of CNN+Pooling operations.
        """         

        with tf.variable_scope("convolution", reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
            input_layer_ = self.input_layer
            for i, num_filter in enumerate(self.config['num_filters']):
                conv_layer = tf.layers.conv2d(inputs=input_layer_,
                                              filters=num_filter,
                                              kernel_size=[self.config['filter_size'][i], self.config['filter_size'][i]],
                                              padding="same",
                                              activation=tf.nn.relu)

                pooling_layer = tf.layers.max_pooling2d(inputs=conv_layer, pool_size=[2, 2], strides=2, padding='same')
                input_layer_ = pooling_layer

            self.model_output_raw = input_layer_

    def build_graph(self, input_layer=None):
        with tf.variable_scope("cnn_model", reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
            if input_layer is None:
                # Here we use RGB modality only.
                self.input_layer = self.input_rgb
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that
            # each frame is considered as a separate sample. We transform
            # [batch_size, seq_len, height, width, num_channels] to [batch_size*seq_len, height, width, num_channels]
            # <op>.shape provides static dimensions that we know at compile-time.
            _, _, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.input_layer = tf.reshape(self.input_layer, [-1, height, width, num_channels])
            self.build_network()

            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]
            batch_seq, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
            self.model_output_flat = tf.reshape(self.model_output_raw, [-1, cnn_height*cnn_width*num_filters])

            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]
            self.model_output_flat = tf.layers.dense(inputs=self.model_output_flat, units=self.config['num_hidden_units'], activation=tf.nn.relu)
            self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, -1, self.config['num_hidden_units']])
            self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)



'''
class C3DModel_RGB(Model):

    def __init__(self, config, placeholders, mode, subsample=1):
        super(C3DModel_RGB, self).__init__(config, placeholders, mode)

        self.input_rgb = placeholders['rgb']

    def build_network(self):
        """
        Stacks convolutional layers where each layer consists of CNN+Pooling operations.
        """
        with tf.variable_scope("3D_conv", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            input_layer_ = self.input_layer

            conv1 = tf.layers.conv3d( inputs=self.input_layer,
                                        filters=16,
                                        kernel_size= [3, 3, 3],
                                        padding="same",
                                        activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            conv1 = tf.nn.leaky_relu (conv1)


            pool1 = tf.layers.max_pooling3d(inputs = conv1, pool_size = [1, 2, 2], strides = [1, 2, 2], padding='same')

            conv2 = tf.layers.conv3d( inputs = pool1,
                                      filters=32,
                                      kernel_size= [3, 3, 3],
                                      padding="same",
                                      activation=None)

            conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
            conv2 = tf.nn.leaky_relu (conv2)

            pool2 = tf.layers.max_pooling3d(inputs = conv2, pool_size = [2, 2, 2], strides = [2, 2, 2], padding='same')

            conv3a = tf.layers.conv3d( inputs = pool2,
                                      filters=64,
                                      kernel_size= [3, 3, 3],
                                      padding="same",
                                      activation=None)

            conv3a = tf.layers.batch_normalization(conv3a, training=self.is_training)
            conv3a = tf.nn.leaky_relu (conv3a)

            conv3b = tf.layers.conv3d( inputs = conv3a,
                                      filters=64,
                                      kernel_size= [3, 3, 3],
                                      padding="same",
                                      activation=None)

            conv3b = tf.layers.batch_normalization(conv3b, training=self.is_training)
            conv3b = tf.nn.leaky_relu (conv3b)

            pool3 = tf.layers.max_pooling3d(inputs = conv3b, pool_size = [2, 2, 2], strides = [2, 2, 2], padding='same')

            conv4a = tf.layers.conv3d( inputs = pool3,
                                      filters = 128,
                                      kernel_size = [3, 3, 3],
                                      padding = "same",
                                      activation = None)

            conv4a = tf.layers.batch_normalization(conv4a, training=self.is_training)
            conv4a = tf.nn.leaky_relu (conv4a)

            conv4b = tf.layers.conv3d( inputs = conv4a,
                                      filters = 128,
                                      kernel_size = [3, 3, 3],
                                      padding = "same",
                                      activation = None)

            conv4b = tf.layers.batch_normalization(conv4b, training=self.is_training)
            conv4b = tf.nn.leaky_relu (conv4b)

            pool4 = tf.layers.max_pooling3d(inputs = conv4b, pool_size = [2, 2, 2], strides = [2, 2, 2], padding='same') 

            conv5a = tf.layers.conv3d( inputs = pool4,
                                      filters = 256,
                                      kernel_size = [3, 3, 3],
                                      padding = "same",
                                      activation = None)

            conv5a = tf.layers.batch_normalization(conv5a, training=self.is_training)
            conv5a = tf.nn.leaky_relu (conv5a)

            conv5b = tf.layers.conv3d( inputs = conv5a,
                                      filters = 256,
                                      kernel_size = [3, 3, 3],
                                      padding = "same",
                                      activation = None)

            conv5b = tf.layers.batch_normalization(conv5b, training=self.is_training)
            conv5b = tf.nn.leaky_relu (conv5b)
            pool5 = tf.layers.average_pooling3d(inputs = conv5b, pool_size = [2, 2, 2], strides = [2, 2, 2], padding='same')
            
            self.model_output_raw = pool5

    def build_graph(self, input_layer=None):
        with tf.variable_scope("C3D_model", reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
            
            if input_layer is None:
                # Here we use RGB modality only.
                self.input_layer = self.input_rgb
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that
            # each frame is considered as a separate sample. We transform
            # [batch_size, seq_len, height, width, num_channels] to [batch_size*seq_len, height, width, num_channels]
            # <op>.shape provides static dimensions that we know at compile-time.
            _, _, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.C3D_Len = self.config['feature_len']

            #self.input_layer = tf.Print (self.input_layer, [tf.shape(self.input_layer)], "Shape: ", summarize = 20)

            n_padding = self.C3D_Len - tf.floormod(self.seq_len_op, self.C3D_Len)
            paddings = [[0, 0], [0, n_padding], [0, 0], [0, 0], [0, 0]]

            self.seq_len_op = self.seq_len_op + n_padding

            self.input_layer = tf.pad (self.input_layer, paddings, "CONSTANT")


            self.input_layer = tf.reshape(self.input_layer, [-1, self.C3D_Len, height, width, num_channels])
            self.build_network()

            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]
            batch_seq, cnn_depth, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
            self.model_output_raw_flat = tf.reshape(self.model_output_raw, [-1, cnn_depth*cnn_height*cnn_width*num_filters])

            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]
            #dropout1 = tf.layers.dropout(inputs=self.model_output_raw_flat, rate=self.config['dropout_rate'], training=self.is_training)

            self.model_output_flat = tf.layers.dense(inputs=self.model_output_raw_flat, units=self.config['num_hidden_units'], activation=None)
            self.model_output_flat = tf.layers.batch_normalization(self.model_output_flat, training=self.is_training)
            self.model_output_flat = tf.nn.leaky_relu (self.model_output_flat)
            self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, -1, self.config['num_hidden_units']])
            self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
'''

class RNNModel(Model):
    """
    Recurrent neural network for sequence modeling.
    - Accepts inputs of rank 3 where a mini-batch has shape of [batch_size, seq_len, feature_size].

    """
    def __init__(self, config, placeholders, mode):
        super(RNNModel, self).__init__(config, placeholders, mode)

        if (config['subsample'] != 1):
            self.input_seq_len = tf.ceil( tf.div (tf.cast(self.input_seq_len, tf.float32), tf.cast(config['subsample'], tf.float32)) )
            self.input_seq_len = tf.cast(self.input_seq_len, tf.int32)

        self.rnn_cell = None
        self.rnn_state = None

    def build_network(self):
        """
        Creates LSTM cell(s) and recurrent model.
        """
        with tf.variable_scope("recurrent", reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
            '''
            rnn_cells = []
            for i in range(self.config['num_layers']):
                rnn_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=self.config['num_hidden_units']))

            if self.config['num_layers'] > 1:
                # Stack multiple cells.
                self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
            else:
                self.rnn_cell = rnn_cells[0]


            self.model_output_raw, self.rnn_state = tf.nn.bidirectional_dynamic_rnn(cell=self.rnn_cell,
                                                                      inputs=self.input_layer,
                                                                      dtype=tf.float32,
                                                                      sequence_length=self.input_seq_len,
                                                                      time_major=False,
                                                                      swap_memory=True)
            '''

            lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=self.config['num_hidden_units'], forget_bias=1.0, state_is_tuple=True)

            lstm_cell_1 = tf.contrib.rnn.AttentionCellWrapper(lstm_cell_1, attn_length = 5, state_is_tuple=True)
            lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, input_keep_prob = 1 - self.config['dropout_rate'],
             output_keep_prob = 1 - self.config['dropout_rate'])

            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=self.config['num_hidden_units'], forget_bias=1.0, state_is_tuple=True)
            lstm_cell_2 = tf.contrib.rnn.AttentionCellWrapper(lstm_cell_2, attn_length = 5, state_is_tuple=True)
            lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, input_keep_prob = 1 - self.config['dropout_rate'],
             output_keep_prob = 1 - self.config['dropout_rate'])

            #lstm_cell_3 = tf.nn.rnn_cell.LSTMCell(num_units=self.config['num_hidden_units'], forget_bias=1.0, state_is_tuple=True)

            lstm_output, self.rnn_state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_1, lstm_cell_2,
                 inputs=self.input_layer, dtype=tf.float32, sequence_length=self.input_seq_len, time_major=False, swap_memory=True)

            self.model_output_raw = tf.concat( [ lstm_output[0], lstm_output[1] ], 2 )

            #self.model_output_raw , self.rnn_state = tf.nn.dynamic_rnn(lstm_cell_3, lstm_output_concat, dtype=tf.float32,
            #                          sequence_length=self.input_seq_len, time_major=False, swap_memory=True)

            self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
            self.saver = tf.train.Saver(self.variable_list)            



    def build_graph(self, input_layer=None):
        with tf.variable_scope("rnn_model", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            if input_layer is None:
                # TODO you can feed any image modality if you wish. You need to flatten images such that a mini-batch
                # has shape [batch_size, seq_len, height*width*num_channels].
                raise Exception("Inputs are missing.")
            else:
                self.input_layer = input_layer

            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]


            self.input_seq_len = tf.floordiv(self.input_seq_len, self.config['feature_len']) + 1
            self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)

            self.build_network()

            
            # Shape of [batch_size, seq_len, representation_size]
            _, _, representation_size = self.model_output_raw.shape.as_list()

            self.model_output = self.model_output_raw
            self.model_output_flat = tf.reshape(self.model_output_raw, [-1, representation_size])



class Skeleton_Encoder(Model):

    def __init__(self, config, placeholders, mode):
        super(Skeleton_Encoder, self).__init__(config, placeholders, mode)

        self.input_sk = placeholders['skeleton_img']

        if (config['subsample'] != 1):
            self.input_seq_len = tf.ceil( tf.div (tf.cast(self.input_seq_len, tf.float32), tf.cast(config['subsample'], tf.float32)) )
            self.input_sk =self.input_sk[:, ::config['subsample'], :, :, :]


        self.saver = None

    def build_network(self):
        """
        Stacks convolutional layers where each layer consists of CNN+Pooling operations.
        """
        def _variable_on_cpu(name, shape, initializer):
            with tf.device('/cpu:0'):
                var = tf.get_variable(name, shape, initializer=initializer)
            return var

        def _variable_with_weight_decay(name, shape, wd):
            var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
            if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return var

        with tf.variable_scope('sk_var', reuse=self.reuse, initializer=self.initializer, regularizer=None) as var_scope:
            self.weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              #'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              #'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
            }
            self.biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              #'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              #'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
            }
            def conv3d(name, l_input, w, b):
                return tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)

            def max_pool(name, l_input, k):
                return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

            input_layer_ = self.input_layer

            # Convolution Layer
            conv1 = conv3d('conv1', input_layer_, self.weights['wc1'], self.biases['bc1'])
            conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            conv1 = tf.nn.relu(conv1, 'relu1')
            pool1 = max_pool('pool1', conv1, k=1)

            # Convolution Layer
            conv2 = conv3d('conv2', pool1, self.weights['wc2'], self.biases['bc2'])
            conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
            conv2 = tf.nn.relu(conv2, 'relu2')
            pool2 = max_pool('pool2', conv2, k=2)

            # Convolution Layer
            conv3 = conv3d('conv3a', pool2, self.weights['wc3a'], self.biases['bc3a'])
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            conv3 = tf.nn.relu(conv3, 'relu3a')
            conv3 = conv3d('conv3b', conv3, self.weights['wc3b'], self.biases['bc3b'])
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            conv3 = tf.nn.relu(conv3, 'relu3b')
            pool3 = max_pool('pool3', conv3, k=2)

            # Convolution Layer
            conv4 = conv3d('conv4a', pool3, self.weights['wc4a'], self.biases['bc4a'])
            conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            conv4 = tf.nn.relu(conv4, 'relu4a')
            conv4 = conv3d('conv4b', conv4, self.weights['wc4b'], self.biases['bc4b'])
            conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            conv4 = tf.nn.relu(conv4, 'relu4b')
            pool4 = max_pool('pool2b', conv4, k=2)

            # Convolution Layer
            #conv5 = conv3d('conv5a', pool4, self.weights['wc5a'], self.biases['bc5a'])
            #conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
            #conv5 = tf.nn.relu(conv5, 'relu5a')
            #conv5 = conv3d('conv5b', conv5, self.weights['wc5b'], self.biases['bc5b'])
            #conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
            #conv5 = tf.nn.relu(conv5, 'relu5b')
            pool5 = max_pool('pool5', pool4, k=2)

            self.model_output_raw = pool5

    def build_graph(self, input_layer=None):
        with tf.variable_scope("C3D_sk_model", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            
            if input_layer is None:
                # Here we use RGB modality only.
                self.input_layer = self.input_sk
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that
            # each frame is considered as a separate sample. We transform
            # [batch_size, seq_len, height, width, num_channels] to [batch_size*seq_len, height, width, num_channels]
            # <op>.shape provides static dimensions that we know at compile-time.
            _, _, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.C3D_Len = self.config['feature_len']


            n_padding = self.C3D_Len - tf.floormod(self.seq_len_op, self.C3D_Len)
            paddings = [[0, 0], [0, n_padding], [0, 0], [0, 0], [0, 0]]

            self.seq_len_op = self.seq_len_op + n_padding

            self.input_layer = tf.pad (self.input_layer, paddings, "CONSTANT")


            self.input_layer = tf.reshape(self.input_layer, [-1, self.C3D_Len, height, width, num_channels])
            self.build_network()

            self.pretrain_var = list ( set(self.weights.values()) | set(self.biases.values()) )
            self.saver = tf.train.Saver(self.pretrain_var)

            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]

            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]
            with tf.variable_scope('c3d_sk_encoder', reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
                batch_seq, cnn_depth, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
                self.model_output_raw_flat = tf.reshape(self.model_output_raw, [-1, cnn_depth*cnn_height*cnn_width*num_filters])

                #dense1 = tf.layers.dense(inputs=self.model_output_raw_flat, units=1024,  activation=None)
                #dense1 = tf.layers.batch_normalization(dense1, training=self.is_training)
                #dense1 = tf.nn.leaky_relu (dense1)
                self.model_output_raw_flat = tf.layers.dropout(inputs=self.model_output_raw_flat, rate=self.config['dropout_rate'], training=self.is_training)
                self.model_output_flat = tf.layers.dense(inputs=self.model_output_raw_flat, units=self.config['sk_num_hidden_units'], activation=None)
                self.model_output_flat = tf.layers.batch_normalization(self.model_output_flat, training=self.is_training)
                self.model_output_flat = tf.nn.leaky_relu (self.model_output_flat)
                
                self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, -1, self.config['sk_num_hidden_units']])

                self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

            self.input_seq_len = tf.floordiv(self.input_seq_len, self.config['feature_len']) + 1
            self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)

class C3DModel_pretrain(Model):

    def __init__(self, config, placeholders, mode):
        super(C3DModel_pretrain, self).__init__(config, placeholders, mode)

        self.input_rgb = placeholders['rgb']

        if (config['subsample'] != 1):
            self.input_seq_len = tf.ceil( tf.div (tf.cast(self.input_seq_len, tf.float32), tf.cast(config['subsample'], tf.float32)) )
            self.input_rgb =self.input_rgb[:, ::config['subsample'], :, :, :]

        self.saver = None

    def build_network(self):
        """
        Stacks convolutional layers where each layer consists of CNN+Pooling operations.
        """
        def _variable_on_cpu(name, shape, initializer):
            with tf.device('/cpu:0'):
                var = tf.get_variable(name, shape, initializer=initializer)
            return var

        def _variable_with_weight_decay(name, shape, wd):
            var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
            if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return var

        with tf.variable_scope('c3d_var', reuse=self.reuse, initializer=self.initializer, regularizer=None) as var_scope:
            self.weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              #'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
              #'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              #'out': _variable_with_weight_decay('wout', [4096, 101], 0.0005)
            }
            self.biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              #'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              #'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              #'out': _variable_with_weight_decay('bout', [101], 0.000),
            }
            def conv3d(name, l_input, w, b):
                return tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)

            def max_pool(name, l_input, k):
                return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

            input_layer_ = self.input_layer

            # Convolution Layer
            conv1 = conv3d('conv1', input_layer_, self.weights['wc1'], self.biases['bc1'])
            conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            conv1 = tf.nn.relu(conv1, 'relu1')
            pool1 = max_pool('pool1', conv1, k=1)

            # Convolution Layer
            conv2 = conv3d('conv2', pool1, self.weights['wc2'], self.biases['bc2'])
            conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
            conv2 = tf.nn.relu(conv2, 'relu2')
            pool2 = max_pool('pool2', conv2, k=2)

            # Convolution Layer
            conv3 = conv3d('conv3a', pool2, self.weights['wc3a'], self.biases['bc3a'])
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            conv3 = tf.nn.relu(conv3, 'relu3a')
            conv3 = conv3d('conv3b', conv3, self.weights['wc3b'], self.biases['bc3b'])
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            conv3 = tf.nn.relu(conv3, 'relu3b')
            pool3 = max_pool('pool3', conv3, k=2)

            # Convolution Layer
            conv4 = conv3d('conv4a', pool3, self.weights['wc4a'], self.biases['bc4a'])
            conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            conv4 = tf.nn.relu(conv4, 'relu4a')
            conv4 = conv3d('conv4b', conv4, self.weights['wc4b'], self.biases['bc4b'])
            conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            conv4 = tf.nn.relu(conv4, 'relu4b')
            pool4 = max_pool('pool4', conv4, k=2)

            # Convolution Layer
            conv5 = conv3d('conv5a', pool4, self.weights['wc5a'], self.biases['bc5a'])
            conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
            conv5 = tf.nn.relu(conv5, 'relu5a')
            conv5 = conv3d('conv5b', conv5, self.weights['wc5b'], self.biases['bc5b'])
            conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
            conv5 = tf.nn.relu(conv5, 'relu5b')
            pool5 = max_pool('pool5', conv5, k=2)

            self.model_output_raw = pool5

    def build_graph(self, input_layer=None):
        with tf.variable_scope("C3D_model", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            
            if input_layer is None:
                # Here we use RGB modality only.
                self.input_layer = self.input_rgb
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that
            # each frame is considered as a separate sample. We transform
            # [batch_size, seq_len, height, width, num_channels] to [batch_size*seq_len, height, width, num_channels]
            # <op>.shape provides static dimensions that we know at compile-time.
            _, _, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.C3D_Len = self.config['feature_len']


            n_padding = self.C3D_Len - tf.floormod(self.seq_len_op, self.C3D_Len)
            paddings = [[0, 0], [0, n_padding], [0, 0], [0, 0], [0, 0]]

            self.seq_len_op = self.seq_len_op + n_padding

            self.input_layer = tf.pad (self.input_layer, paddings, "CONSTANT")


            self.input_layer = tf.reshape(self.input_layer, [-1, self.C3D_Len, height, width, num_channels])
            self.build_network()

            self.pretrain_var = list ( set(self.weights.values()) | set(self.biases.values()) )
            self.saver = tf.train.Saver(self.pretrain_var)

            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]

            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]
            with tf.variable_scope('c3d_rgb_encoder', reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
                batch_seq, cnn_depth, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
                self.model_output_raw_flat = tf.reshape(self.model_output_raw, [-1, cnn_depth*cnn_height*cnn_width*num_filters])

                #dense1 = tf.layers.dense(inputs=self.model_output_raw_flat, units=1024,  activation=None)
                #dense1 = tf.layers.batch_normalization(dense1, training=self.is_training)
                #dense1 = tf.nn.leaky_relu (dense1)
                self.model_output_raw_flat = tf.layers.dropout(inputs=self.model_output_raw_flat, rate=self.config['dropout_rate'], training=self.is_training)
                self.model_output_flat = tf.layers.dense(inputs=self.model_output_raw_flat, units=self.config['rgb_num_hidden_units'], activation=None)
                self.model_output_flat = tf.layers.batch_normalization(self.model_output_flat, training=self.is_training)
                self.model_output_flat = tf.nn.leaky_relu (self.model_output_flat)
                
                self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, -1, self.config['rgb_num_hidden_units']])

                self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

            self.input_seq_len = tf.floordiv(self.input_seq_len, self.config['feature_len']) + 1
            self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)

class C3D_Depth(Model):

    def __init__(self, config, placeholders, mode):
        super(C3D_Depth, self).__init__(config, placeholders, mode)

        self.input_depth = tf.image.grayscale_to_rgb(placeholders['depth'] / 255.0)
        self.input_mask = placeholders['segmentation'] / 255.0
        self.input_processed = tf.multiply(self.input_depth, self.input_mask)

        if (config['subsample'] != 1):
            self.input_seq_len = tf.ceil( tf.div (tf.cast(self.input_seq_len, tf.float32), tf.cast(config['subsample'], tf.float32)) )
            self.input_processed =self.input_processed[:, ::config['subsample'], :, :, :]
        self.saver = None

    def build_network(self):
        """
        Stacks convolutional layers where each layer consists of CNN+Pooling operations.
        """
        def _variable_on_cpu(name, shape, initializer):
            with tf.device('/cpu:0'):
                var = tf.get_variable(name, shape, initializer=initializer)
            return var

        def _variable_with_weight_decay(name, shape, wd):
            var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
            if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return var

        with tf.variable_scope('c3d_depth', reuse=self.reuse, initializer=self.initializer, regularizer=None) as var_scope:
            self.weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              #'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              #'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              #'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
              #'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              #'out': _variable_with_weight_decay('wout', [4096, 101], 0.0005)
            }
            self.biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              #'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              #'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              #'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              #'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              #'out': _variable_with_weight_decay('bout', [101], 0.000),
            }
            def conv3d(name, l_input, w, b):
                return tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)

            def max_pool(name, l_input, k):
                return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

            input_layer_ = self.input_layer

            # Convolution Layer
            conv1 = conv3d('conv1', input_layer_, self.weights['wc1'], self.biases['bc1'])
            conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
            conv1 = tf.nn.relu(conv1, 'relu1')
            pool1 = max_pool('pool1', conv1, k=1)

            # Convolution Layer
            conv2 = conv3d('conv2', pool1, self.weights['wc2'], self.biases['bc2'])
            conv2 = tf.layers.batch_normalization(conv2, training=self.is_training)
            conv2 = tf.nn.relu(conv2, 'relu2')
            pool2 = max_pool('pool2', conv2, k=2)

            # Convolution Layer
            conv3 = conv3d('conv3a', pool2, self.weights['wc3a'], self.biases['bc3a'])
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            conv3 = tf.nn.relu(conv3, 'relu3a')
            conv3 = conv3d('conv3b', conv3, self.weights['wc3b'], self.biases['bc3b'])
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_training)
            conv3 = tf.nn.relu(conv3, 'relu3b')
            pool3 = max_pool('pool3', conv3, k=2)

            # Convolution Layer
            conv4 = conv3d('conv4a', pool3, self.weights['wc4a'], self.biases['bc4a'])
            conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            conv4 = tf.nn.relu(conv4, 'relu4a')
            conv4 = conv3d('conv4b', conv4, self.weights['wc4b'], self.biases['bc4b'])
            conv4 = tf.layers.batch_normalization(conv4, training=self.is_training)
            conv4 = tf.nn.relu(conv4, 'relu4b')
            pool4 = max_pool('pool4', conv4, k=2)

            # Convolution Layer
            #conv5 = conv3d('conv5a', pool4, self.weights['wc5a'], self.biases['bc5a'])
            #conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
            #conv5 = tf.nn.relu(conv5, 'relu5a')
            #conv5 = conv3d('conv5b', conv5, self.weights['wc5b'], self.biases['bc5b'])
            #conv5 = tf.layers.batch_normalization(conv5, training=self.is_training)
            #conv5 = tf.nn.relu(conv5, 'relu5b')
            pool5 = max_pool('pool5', pool4, k=2)

            self.model_output_raw = pool5

    def build_graph(self, input_layer=None):
        with tf.variable_scope("C3D_depth", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            
            if input_layer is None:
                self.input_layer = self.input_processed
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that
            # each frame is considered as a separate sample. We transform
            # [batch_size, seq_len, height, width, num_channels] to [batch_size*seq_len, height, width, num_channels]
            # <op>.shape provides static dimensions that we know at compile-time.
            _, _, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.C3D_Len = self.config['feature_len']


            n_padding = self.C3D_Len - tf.floormod(self.seq_len_op, self.C3D_Len)
            paddings = [[0, 0], [0, n_padding], [0, 0], [0, 0], [0, 0]]

            self.seq_len_op = self.seq_len_op + n_padding

            self.input_layer = tf.pad (self.input_layer, paddings, "CONSTANT")


            self.input_layer = tf.reshape(self.input_layer, [-1, self.C3D_Len, height, width, num_channels])
            self.build_network()

            self.pretrain_var = list ( set(self.weights.values()) | set(self.biases.values()) )
            self.saver = tf.train.Saver(self.pretrain_var)

            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]


            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]

            with tf.variable_scope('c3d_depth_encoder', reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
                batch_seq, cnn_depth, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
                self.model_output_raw_flat = tf.reshape(self.model_output_raw, [-1, cnn_depth*cnn_height*cnn_width*num_filters])
                #dense1 = tf.layers.dense(inputs=self.model_output_raw_flat, units=1024,  activation=None)
                #dense1 = tf.layers.batch_normalization(dense1, training=self.is_training)
                #dense1 = tf.nn.leaky_relu (dense1)
                self.model_output_raw_flat = tf.layers.dropout(inputs=self.model_output_raw_flat, rate=self.config['dropout_rate'], training=self.is_training)
                self.model_output_flat = tf.layers.dense(inputs=self.model_output_raw_flat, units=self.config['depth_num_hidden_units'], activation=None)
                self.model_output_flat = tf.layers.batch_normalization(self.model_output_flat, training=self.is_training)
                self.model_output_flat = tf.nn.leaky_relu (self.model_output_flat)

                self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, -1, self.config['depth_num_hidden_units']])                
                self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)


            self.input_seq_len = tf.floordiv(self.input_seq_len, self.config['feature_len']) + 1
            self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)




'''

class R3D_rgb(Model):

    def __init__(self, config, placeholders, mode):
        super(R3D_rgb, self).__init__(config, placeholders, mode)

        self.input_rgb = placeholders['rgb']

        self.saver = None

    def build_network(self):

        def identity_block(input_tensor, kernel_size, filters):
            """The identity block is the block that has no conv layer at shortcut.
            # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            """
            filters1, filters2 = filters
            bn_axis = 4
 
            x = tf.keras.layers.Conv3D(filters1, (1, kernel_size, kernel_size), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(input_tensor)

            x = tf.keras.layers.Conv3D(filters1, (kernel_size, 1, 1),  padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

            x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
            x = tf.keras.layers.Activation('relu')(x)
 
            x = tf.keras.layers.Conv3D(filters2, (1, kernel_size, kernel_size), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

            x = tf.keras.layers.Conv3D(filters2, (kernel_size, 1, 1), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

            x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
 
            x = tf.keras.layers.add([x, input_tensor])
            x = tf.keras.layers.Activation('relu')(x)
            return x

        def conv_block(input_tensor, kernel_size, filters, strides):
            """A block that has a conv layer at shortcut.
            # Arguments
                input_tensor: input tensor
                kernel_size: default 3, the kernel size of middle conv layer at main path
                filters: list of integers, the filters of 3 conv layer at main path
                stage: integer, current stage label, used for generating layer names
            # Returns
                Output tensor for the block.
            Note that from stage 3,
            the second conv layer at main path is with strides=(2, 2)
            And the shortcut should have strides=(2, 2) as well
            """
 
            filters1, filters2 = filters
            bn_axis = 4

            x = tf.keras.layers.Conv3D(filters1, (1, kernel_size, kernel_size), strides=(1, strides, strides), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(input_tensor)

            x = tf.keras.layers.Conv3D(filters1, (kernel_size, 1, 1), strides=(strides, 1, 1), padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x)

            x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)

            x = tf.keras.layers.Activation('relu')(x)
 
            x = tf.keras.layers.Conv3D(filters2, (1, kernel_size, kernel_size), strides=(1, 1, 1),
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

            x = tf.keras.layers.Conv3D(filters2, (kernel_size, 1, 1), strides=(1, 1, 1),
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

            x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
 
            shortcut = tf.keras.layers.Conv3D(filters2, (1, 1, 1), strides=(strides, strides, strides),
                             use_bias=False,
                             kernel_initializer='he_normal')(input_tensor)

            shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis)(shortcut)
 
            x = tf.keras.layers.add([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            return x


        with tf.variable_scope('r3d_resnet', reuse=self.reuse, initializer=self.initializer, regularizer=None) as var_scope:
           
            input_layer_ = self.input_layer

            bn_axis = 4
 
            # Conv1 (7x7,64,stride=(1,2,2)) 16x80x80
 
            x = tf.keras.layers.Conv3D(64, (3, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False,
                      kernel_initializer='he_normal')(input_layer_)

            x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
            x = tf.keras.layers.Activation('relu')(x)


            # Conv2_x 16x40x40

            x = conv_block(x, 3, [32, 32], strides=2)
            x = identity_block(x, 3, [32, 32])

            # Conv3_x 8x20x20

            x = conv_block(x, 3, [64, 64], strides=2)
            x = identity_block(x, 3, [64, 64])

            # Conv4_x 4x10x10

            x = conv_block(x, 3, [128, 128], strides=2)
            x = identity_block(x, 3, [128, 128])
  
            # Conv5_x 2x5x5

            x = conv_block(x, 3, [256, 256], strides=2)
            x = identity_block(x, 3, [256, 256])

            #out 1x3x3

            return x
    
    def build_graph(self, input_layer=None):
        with tf.variable_scope("R3D_RGB", reuse=self.reuse, initializer=self.initializer, regularizer=None) as vs:
            
            if input_layer is None:
                self.input_layer = self.input_processed
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that
            # each frame is considered as a separate sample. We transform
            # [batch_size, seq_len, height, width, num_channels] to [batch_size*seq_len, height, width, num_channels]
            # <op>.shape provides static dimensions that we know at compile-time.
            _, _, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.C3D_Len = self.config['feature_len']


            n_padding = self.C3D_Len - tf.floormod(self.seq_len_op, self.C3D_Len)
            paddings = [[0, 0], [0, n_padding], [0, 0], [0, 0], [0, 0]]

            self.seq_len_op = self.seq_len_op + n_padding

            self.input_layer = tf.pad (self.input_layer, paddings, "CONSTANT")


            self.input_layer = tf.reshape(self.input_layer, [-1, self.C3D_Len, height, width, num_channels])
            
            self.model_output_raw = self.build_network()


            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]


            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]

            with tf.variable_scope('r3d_rgb_encoder', reuse=self.reuse, initializer=self.initializer, regularizer=None):
                batch_seq, cnn_depth, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
                self.model_output_raw_flat = tf.reshape(self.model_output_raw, [-1, cnn_depth*cnn_height*cnn_width*num_filters])
                #dropout1 = tf.layers.dropout(inputs=self.model_output_raw_flat, rate=self.config['dropout_rate'], training=self.is_training)
                #dense1 = tf.layers.dense(inputs=dropout1, units=1024,  activation=tf.nn.leaky_relu)
                #dropout2 = tf.layers.dropout(inputs=dense1, rate=self.config['dropout_rate'], training=self.is_training)
                self.model_output_flat = tf.layers.dense(inputs=self.model_output_raw_flat, units=self.config['num_hidden_units'])
                self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, -1, self.config['num_hidden_units']])
                
            
            self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

            self.input_seq_len = tf.floordiv(self.input_seq_len, self.config['feature_len']) + 1
            self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)
'''
