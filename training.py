"""Copyright (c) 2019 AIT Lab, ETH Zurich, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import os
import time
import json

import tensorflow as tf
from dataset import TFRecordDataset
from model import *


"""
Training script loading data, creating model and running training. You will see some TODO statements in the code. 
Please note that they are just the best practices and do not guarantee a better performance. Hence, you may ignore
them.  
"""

def _variable_summaries(var, name):
    with tf.name_scope('summaries_' + str(name)):
        mean = tf.reduce_mean(var)
        s_mean = tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        s_std = tf.summary.scalar('stddev', stddev)
        s_max = tf.summary.scalar('max', tf.reduce_max(var))
        s_min = tf.summary.scalar('min', tf.reduce_min(var))
        sum_var = {'mean': s_mean, 'std':s_std, 'max': s_max, 'min': s_min}
    return sum_var

def main(config):
    # TODO
    # Here you can call your preprocessing functions. If you generate intermediate representations, you should be
    # using config['tmp_dir'] directory.
    # If you use a different training/validation split than what we provide, please make sure that this split is
    # reproducible. You can either set `seed` or save the split indices into a  file and submit it along with your code.

    #############
    # Data
    #############

    # Each <key,value> pair in `training_placeholders` and `validation_placeholders` corresponds to a TF placeholder.
    # Create input placeholders for training data.
    
    #training_file_pattern = os.path.join(config['data_dir'], "training-??-of-??")
    training_file_pattern = [os.path.join(config['data_dir'], "training-??-of-??"), os.path.join(config['data_dir'], "validation-??-of-??")]

    training_dataset = TFRecordDataset(data_path=training_file_pattern,
                                       batch_size=config['batch_size'],
                                       shuffle=True,
                                       normalize=config['normalize_data'])
    training_iterator = training_dataset.get_iterator()
    training_placeholders = training_dataset.get_tf_samples()

    # Create input placeholders for validation data.
    validation_file_pattern = os.path.join(config['data_dir'], "validation-??-of-??")
    #validation_file_pattern = os.path.join(config['data_dir'], "test-??-of-??")

    validation_dataset = TFRecordDataset(data_path=validation_file_pattern,
                                         batch_size=config['batch_size'],
                                         shuffle=False,
                                         normalize=config['normalize_data'])
    validation_iterator = validation_dataset.get_iterator()
    validation_placeholders = validation_dataset.get_tf_samples()

    # Using RGB modality.
    training_input_layer = training_placeholders['rgb']
    validation_input_layer = validation_placeholders['rgb']

    ##################
    # Training Model
    ##################
    # Create separate graphs for training and validation.
    # Training graph.
    with tf.name_scope("Training"):
        # Create model
        c3d_model = C3DModel_pretrain(config=config['c3d'],
                             placeholders=training_placeholders,
                             mode='training')
        c3d_model.build_graph()

        c3d_depth = C3D_Depth(config=config['c3d'],
                             placeholders=training_placeholders,
                             mode='training')
        c3d_depth.build_graph()

        sk_encoder = Skeleton_Encoder(config=config['c3d'],
                             placeholders=training_placeholders,
                             mode='training')
        sk_encoder.build_graph()

        feature = tf.concat([c3d_model.model_output, c3d_depth.model_output, sk_encoder.model_output], axis=2)

        train_model = RNNModel(config=config['rnn'],
                               placeholders=training_placeholders,
                               mode="training")
        train_model.build_graph(input_layer=feature)
        train_model.build_loss()

        print("\n# of parameters: %s"%train_model.get_num_parameters())

        ##############
        # Optimization
        ##############
        global_step = tf.Variable(1, name='global_step', trainable=False)
        if config['learning_rate_type'] == 'exponential':
            learning_rate = tf.train.exponential_decay(config['learning_rate'],
                                                       global_step=global_step,
                                                       decay_steps=500,
                                                       decay_rate=0.97,
                                                       staircase=False)
        elif config['learning_rate_type'] == 'fixed':
            learning_rate = config['learning_rate']
        else:
            raise Exception("Invalid learning rate type")

        '''
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(train_model.loss, global_step=global_step)
        '''
        var_list = c3d_model.variable_list + c3d_depth.variable_list + sk_encoder.variable_list + train_model.variable_list
        pretrain_var_list = c3d_depth.pretrain_var + c3d_model.pretrain_var + sk_encoder.pretrain_var
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):
            pretrain_opt = tf.train.AdamOptimizer(config['pretrain_lr'] ).minimize(train_model.loss, global_step=global_step, var_list=pretrain_var_list)
            train_opt = tf.train.AdamOptimizer(learning_rate).minimize(train_model.loss, global_step=global_step, var_list=var_list)



    ###################
    # Validation Model
    ###################
    with tf.name_scope("Validation"):
        # Create model
        valid_c3d_model = C3DModel_pretrain(config=config['c3d'],
                                   placeholders=validation_placeholders,
                                   mode='validation')
        valid_c3d_model.build_graph()

        valid_c3d_depth = C3D_Depth(config=config['c3d'],
                             placeholders=validation_placeholders,
                             mode='validation')
        valid_c3d_depth.build_graph()

        valid_sk_encoder = Skeleton_Encoder(config=config['c3d'],
                             placeholders=validation_placeholders,
                             mode='validation')
        valid_sk_encoder.build_graph()

        valid_feature = tf.concat([valid_c3d_model.model_output, valid_c3d_depth.model_output, valid_sk_encoder.model_output], axis=2)

        valid_model = RNNModel(config=config['rnn'],
                               placeholders=validation_placeholders,
                               mode="validation")

        valid_model.build_graph(input_layer=valid_feature)
        valid_model.build_loss()

    ##############
    # Monitoring
    ##############
    # Create placeholders to provide tensorflow average loss and accuracy.
    loss_avg_pl = tf.placeholder(tf.float32, name="loss_avg_pl")
    accuracy_avg_pl = tf.placeholder(tf.float32, name="accuracy_avg_pl")

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and plots evaluation results.
    summary_train_loss = tf.summary.scalar('loss', train_model.loss)
    summary_train_acc = tf.summary.scalar('accuracy_training', train_model.batch_accuracy)
    summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg_pl)
    summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg_pl)
    summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('summary_var'):
        sk_sum = _variable_summaries(sk_encoder.variable_list[0], 'sk')
        wc4_sum = _variable_summaries(c3d_model.weights['wc4a'], 'wc4a')
        depth_sum = _variable_summaries(c3d_depth.variable_list[0], 'depth')
        rnn_sum = _variable_summaries(train_model.variable_list[0], 'rnn')

    # Group summaries. summaries_training is used during training and reported after every step.
    summaries_training = tf.summary.merge([summary_train_loss, summary_train_acc, summary_learning_rate,
                                           wc4_sum['mean'], wc4_sum['max'], rnn_sum['mean'], rnn_sum['max'],
                                        sk_sum['mean'], sk_sum['max'], depth_sum['mean'], depth_sum['max']])

    #summaries_training = tf.summary.merge([summary_train_loss, summary_train_acc, summary_learning_rate])
    # summaries_evaluation is used by both training and validation in order to report the performance on the dataset.
    summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

    # Create session object
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    # Add the ops to initialize variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually initialize the variables
    session.run(init_op)



    # Register summary ops.
    train_summary_dir = os.path.join(config['model_dir'], "summary", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)
    valid_summary_dir = os.path.join(config['model_dir'], "summary", "valid")
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, session.graph)

    # Create a saver for saving checkpoints.

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10, save_relative_paths=True)

    if (config['finetune']):
        saver.restore(session, config['pretrained'])
    else:
        if (config['c3d']['pretrain_rgb'] is not None):
            c3d_model.saver.restore(session, config['c3d']['pretrain_rgb'])
        if (config['c3d']['pretrain_depth'] is not None):
            c3d_depth.saver.restore(session, config['c3d']['pretrain_depth'])
        if (config['c3d']['pretrain_sk'] is not None):
            sk_encoder.saver.restore(session, config['c3d']['pretrain_sk'])

    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = 0.0
    counter_loss_training = 0.0
    counter_correct_predictions_validation = 0.0
    counter_loss_validation = 0.0

    # Save configuration in json formats.
    json.dump(config, open(os.path.join(config['model_dir'], 'config.json'), 'w'), indent=4, sort_keys=True)

    ##########################
    # Training Loop
    ##########################
    session.run(training_iterator.initializer)
    session.run(validation_iterator.initializer)
    epoch = 0
    step = 0
    stop_signal = False
    while not stop_signal:
        # Run training for some steps and then run evaluation on the validation split.
        for i in range(config['evaluate_every_step']):
            try:
                step += 1
                start_time = time.perf_counter()
                # Run the optimizer to update weights.
                # Note that "train_op" is responsible from updating network weights.
                # Only the operations that are fed are evaluated.
                # Run the optimizer to update weights.
                train_summary, num_correct_predictions, loss, _, _ = session.run([summaries_training,
                                                                               train_model.num_correct_predictions,
                                                                               train_model.loss,
                                                                               train_opt, pretrain_opt],
                                                                              feed_dict={})
                # Update counters.
                counter_correct_predictions_training += num_correct_predictions
                counter_loss_training += loss
                # Write summary data.
                if (step%10) == 0:
                    train_summary_writer.add_summary(train_summary, step)

                # Report training performance
                if (step%config['print_every_step']) == 0:
                    # To get a smoother loss plot, we calculate average performance.
                    accuracy_avg = counter_correct_predictions_training/(
                                config['batch_size']*config['print_every_step'])
                    loss_avg = counter_loss_training/(config['print_every_step'])
                    # Feed average performance.
                    summary_report, out_feature = session.run([summaries_evaluation, feature],
                                                 feed_dict={accuracy_avg_pl: accuracy_avg, loss_avg_pl: loss_avg})
                    train_summary_writer.add_summary(summary_report, step)
                    time_elapsed = (time.perf_counter() - start_time)/config['print_every_step']
                    print("[Train/%d] Accuracy: %.3f, Loss: %.3f, time/step = %.3f"%(step,
                                                                                     accuracy_avg,
                                                                                     loss_avg,
                                                                                     time_elapsed))
                    # print(out_feature.shape)
                    # print(out_feature[0][0])


                    counter_correct_predictions_training = 0.0
                    counter_loss_training = 0.0

            except tf.errors.OutOfRangeError:
                # Dataset iterator throws an exception after all samples are used (i.e., epoch). Reinitialize the
                # iterator to start a new epoch.
                session.run(training_iterator.initializer)
                epoch += 1
                if epoch >= config['num_epochs']:
                    stop_signal = True
                    break

        # Evaluate the model.
        validation_step = 0
        start_time = time.perf_counter()
        try:
            # Here we evaluate our model on entire validation split.
            # Validation model fetches the data samples from the iterator every session.run that evaluates an
            # op using input sample. We don't need to do anything else.
            while True:
                # Calculate average validation accuracy.
                num_correct_predictions, loss = session.run([valid_model.num_correct_predictions,
                                                             valid_model.loss])
                # Update counters.
                counter_correct_predictions_validation += num_correct_predictions
                counter_loss_validation += loss
                validation_step += 1

        except tf.errors.OutOfRangeError:
            # Report validation performance
            accuracy_avg = counter_correct_predictions_validation/(config['batch_size']*validation_step)
            loss_avg = counter_loss_validation/validation_step
            summary_report = session.run(summaries_evaluation,
                                         feed_dict={accuracy_avg_pl: accuracy_avg, loss_avg_pl: loss_avg})
            valid_summary_writer.add_summary(summary_report, step)
            time_elapsed = (time.perf_counter() - start_time)/validation_step
            print("[Valid/%d] Accuracy: %.3f, Loss: %.3f, time/step = %.3f"%(step,
                                                                             accuracy_avg,
                                                                             loss_avg,
                                                                             time_elapsed))

            counter_correct_predictions_validation = 0.0
            counter_loss_validation = 0.0
            # Initialize the validation data iterator for the next evaluation round.
            session.run(validation_iterator.initializer)

        # TODO
        # You can implement early stopping by using the validation performance calculated above. If it doesn't improve
        # for a certain number of "tolerance" steps, you can stop the training. You can also save a checkpoint only
        # if there is an improvement on the validation performance. Since monitoring and controlling the overfitting is
        # one of the major problems of this project, I advise you to invest some time on early stopping. Moreover,
        # you no longer need to worry about the number of training epochs hyper-parameter.
        # Hint: keep track of the best validation loss or accuracy. if it doesn't satisfy your criteria, set True to
        # the stop_signal.

        if (step % config['checkpoint_every_step']) == 0:
            ckpt_save_path = saver.save(session, os.path.join(config['model_dir'], 'model'), global_step)
            print("Model saved in file: %s" % ckpt_save_path)

    session.close()
    # Evaluate model after training and create submission file.
    tf.reset_default_graph()
    from restore_and_evaluate import main as evaluate
    config['checkpoint_id'] = None
    evaluate(config)

    # TODO
    # After you found the best performing hyper-parameters on the validation split, you can train the model by using
    # both the training and validation splits.


if __name__ == '__main__':
    from config import config
    main(config)
