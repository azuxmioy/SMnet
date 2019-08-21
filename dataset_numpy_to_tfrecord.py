"""Copyright (c) 2019 AIT Lab, ETH Zurich, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import numpy as np
import tensorflow as tf

"""
This is an auxiliary code for storing data in tfrecords that is required by the model training/evaluation code.

If you can't implement your custom preprocessing routines in Tensorflow, you can first convert the given tfrecord
data to numpy, apply preprocessing in python/numpy and finally store in tfrecords. If you do so, make sure that it is
reproducible.
"""

RNG = np.random.RandomState(42)


def create_tfrecord_writers(output_file, n_shards):
    writers = []
    for i in range(1, n_shards+1):
        tf_writer_options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
        writers.append(tf.python_io.TFRecordWriter("{}-{:0>2d}-of-{:0>2d}".format(output_file, i, n_shards), options=tf_writer_options))
    return writers


def close_tfrecord_writers(writers):
    for w in writers:
        w.close()


def write_tfexample(writers, tf_example):
    random_writer_idx = RNG.randint(0, len(writers))
    writers[random_writer_idx].write(tf_example.SerializeToString())


def to_tfexample(sample):
    features = dict()
    features['rgb'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample["rgb"].tobytes()]))
    features['depth'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample["depth"].tobytes()]))
    features['segmentation'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample["segmentation"].tobytes()]))
    features['skeleton'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample["skeleton"].tobytes()]))
    features['skeleton_img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample["skeleton_img"].tobytes()]))
    features['length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([sample["rgb"].shape[0]])))
    features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([sample["label"]])))
    features['id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([sample["id"]])))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

'''
# List of sample dictionaries where each sample dictionary contains rgb, depth, segmentation, skeleton, label and
# id fields (see the sample parameter of to_tfexample function).
samples = []
output_dir = ""  # Output directory.
n_shards = 20
tfrecord_writers = create_tfrecord_writers(output_dir, n_shards)
for data_sample in samples:
    tfexample = to_tfexample(data_sample)
    write_tfexample(tfrecord_writers, tfexample)
close_tfrecord_writers(tfrecord_writers)
'''
