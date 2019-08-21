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
import os
import glob
import argparse
import json

from dataset import TFRecordDataset
from model import *
from utils import create_submission_files


def main(config, args):
    # Create input placeholders for test data.
    test_file_pattern = os.path.join(config['data_dir'], "test-??-of-??")
    test_dataset = TFRecordDataset(data_path=test_file_pattern,
                                   batch_size=config['batch_size'],
                                   shuffle=False,
                                   normalize=config['normalize_data'])
    test_iterator = test_dataset.get_iterator()
    test_placeholders = test_dataset.get_tf_samples()

    # Using RGB modality.
    test_input_layer = test_placeholders['rgb']

    session = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    # Test graph.
    with tf.name_scope("Test"):
        # Create model
        '''
        test_cnn_model = C3DModel_pretrain(config=config['c3d'],
                                  placeholders=test_placeholders,
                                  mode='test')
        test_cnn_model.build_graph(input_layer=test_input_layer)

        test_model = RNNModel(config=config['rnn'],
                              placeholders=test_placeholders,
                              mode="test")
        test_model.build_graph(input_layer=test_cnn_model.model_output)
        test_model.build_loss()
        '''
        test_c3d_model = C3DModel_pretrain(config=config['c3d'],
                                   placeholders=test_placeholders,
                                   mode='test')
        test_c3d_model.build_graph()

        test_c3d_depth = C3D_Depth(config=config['c3d'],
                             placeholders=test_placeholders,
                             mode='test')
        test_c3d_depth.build_graph()

        test_sk_encoder = Skeleton_Encoder(config=config['c3d'],
                             placeholders=test_placeholders,
                             mode='test')
        test_sk_encoder.build_graph()

        test_feature = tf.concat([test_c3d_model.model_output, test_c3d_depth.model_output, test_sk_encoder.model_output], axis=2)


        test_model = RNNModel(config=config['rnn'],
                               placeholders=test_placeholders,
                               mode="test")

        test_model.build_graph(input_layer=test_feature)
        test_model.build_loss()

    # Restore computation graph.
    saver = tf.train.Saver(save_relative_paths=True)
    # Restore variables.
    checkpoint_path = config['checkpoint_id']
    if checkpoint_path is None:
        checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
    else:
        pass
    print("Evaluating " + checkpoint_path)
    saver.restore(session, checkpoint_path)

    # Evaluation loop
    test_predictions = []
    test_softlabels = []
    test_sample_ids = []
    session.run(test_iterator.initializer)
    try:
        while True:
            # Get predicted labels and sample ids for submission csv.
            [predictions, softlabel, sample_ids] = session.run([test_model.predictions, test_model.softlabel, test_placeholders['id']], feed_dict={})
            test_predictions.extend(predictions)
            test_softlabels.extend(softlabel)
            test_sample_ids.extend(sample_ids)

    except tf.errors.OutOfRangeError:
        print('Done.')

    # Writes submission file.
    sorted_labels = [label for _, label in sorted(zip(test_sample_ids, test_predictions))]
    sorted_softlabels = [softlabel for _, softlabel in sorted(zip(test_sample_ids, test_softlabels))]
    create_submission_files(labels=sorted_labels, soft_labels=sorted_softlabels,
                            out_dir=config['model_dir'],
                            out_csv_file=config['model_id'] + '_'+ str(args.checkpoint_id) + '_submission.csv',
                            out_code_file=config['model_id'] + '_'+ str(args.checkpoint_id) +  '_code.zip',
                            out_soft_file=config['model_id'] + '_'+ str(args.checkpoint_id) +  '_softlabel.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save_dir', dest='save_dir', type=str, default='./runs/', help='path to main model save directory')
    parser.add_argument('-M', '--model_id', dest='model_id', type=str, help='10-digit model id, i.e., timestamp')
    parser.add_argument('-C', '--checkpoint_id', type=str, default=None, help='checkpoint id (only step number)')
    args = parser.parse_args()

    try:
        experiment_dir = glob.glob(os.path.join(args.save_dir, args.model_id + "*"), recursive=False)[0]
    except IndexError:
        raise Exception("Model " + str(args.model_id) + " is not found in " + str(args.save_dir))

    # Loads config file from experiment folder.
    config = json.load(open(os.path.abspath(os.path.join(experiment_dir, 'config.json')), 'r'))
    if args.checkpoint_id is not None:
        config['checkpoint_id'] = os.path.join(experiment_dir, 'model-' + str(args.checkpoint_id))
    else:
        config['checkpoint_id'] = None  # The latest checkpoint will be used.

    main(config, args)
