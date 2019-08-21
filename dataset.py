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
import numpy as np
import functools
import os
import dataset_numpy_to_tfrecord as n2t

class TFRecordDataset:
    """
    Dataset class for reading TFRecord files. You can implement
    """
    def __init__(self, data_path, batch_size, normalize=False, shuffle=True, num_parallel_calls=4, **kwargs):
        # To reshape the serialized data. Do not change these values unless you create tfrecords yourself and have
        # different size.
        self.RGB_SIZE = (-1, 80, 80, 3)
        self.DEPTH_SIZE = (-1, 80, 80, 1)
        self.BINARY_SIZE = (-1, 80, 80, 3)
        self.SKELETON_SIZE = (-1, 180)

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_parallel_calls = num_parallel_calls
        self.normalize = normalize
        self.tf_data = None

        self.tf_data_transformations()
        self.tf_data_to_model()

        if tf.executing_eagerly():
            self.iterator = self.tf_data.make_one_shot_iterator()
            self.tf_samples = None
        else:
            self.iterator = self.tf_data.make_initializable_iterator()
            self.tf_samples = self.iterator.get_next()

    def get_iterator(self):
        return self.iterator

    def get_tf_samples(self):
        return self.tf_samples

    def tf_data_transformations(self):
        """
        Loads the raw data and apply preprocessing.
        This method is also used in calculation of the dataset statistics (i.e., meta-data file).
        """
        tf_data_opt = tf.data.Options()
        tf_data_opt.experimental_autotune = True

        tf_data_files = tf.data.Dataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
        self.tf_data = tf.data.TFRecordDataset(filenames=tf_data_files, compression_type="ZLIB", num_parallel_reads=self.num_parallel_calls)
        self.tf_data = self.tf_data.with_options(tf_data_opt)


        self.tf_data = self.tf_data.map(functools.partial(self.__parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.prefetch(self.batch_size*20)

        
        if self.shuffle:
            self.tf_data = self.tf_data.shuffle(1000)
        if self.normalize:
            self.tf_data = self.tf_data.map(functools.partial(self.__normalize_with_local_stats),
                                            num_parallel_calls=self.num_parallel_calls)

    def tf_data_to_model(self):
        self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=self.tf_data.output_shapes)
        self.tf_data = self.tf_data.prefetch(2)

    def __normalize_with_local_stats(self, tf_sample_dict):
        """
        Given a sample dictionary (see __parse_single_tfexample_fn return), calculates mean and std and applies
        zero-mean, unit-variance standardization.
        """
        def get_mean_and_std(tensor, keepdims=False):
            """
            Calculates mean and standard deviation of a tensor over given dimensions.
            """
            mean = tf.reduce_mean(tensor, keepdims=True)
            diff_squared = tf.square(tensor - mean)
            variance = tf.reduce_mean(diff_squared, keepdims=keepdims)
            std = tf.maximum(tf.sqrt(variance), 1e-6)
            return mean, std

        rgb_mean, rgb_std = get_mean_and_std(tf_sample_dict["rgb"])
        sk_mean, sk_std = get_mean_and_std(tf_sample_dict["skeleton_img"])

        tf_sample_dict["rgb"] = (tf_sample_dict["rgb"] - rgb_mean) / rgb_std
        tf_sample_dict["skeleton_img"] = (tf_sample_dict["skeleton_img"] - sk_mean) / sk_std

        return tf_sample_dict

    def __parse_single_tfexample_fn(self, proto):
        feature_to_type = {
            "rgb"         : tf.FixedLenFeature([], dtype=tf.string),
            "depth"       : tf.FixedLenFeature([], dtype=tf.string),
            "segmentation": tf.FixedLenFeature([], dtype=tf.string),
            "skeleton"    : tf.FixedLenFeature([], dtype=tf.string),
            "skeleton_img": tf.FixedLenFeature([], dtype=tf.string),
            "length"      : tf.FixedLenFeature([1], dtype=tf.int64),
            "label"       : tf.FixedLenFeature([1], dtype=tf.int64),
            "id"          : tf.FixedLenFeature([1], dtype=tf.int64),
            }

        features = tf.parse_single_example(proto, feature_to_type)
        features["rgb"] = tf.reshape(tf.decode_raw(features['rgb'], tf.float32), self.RGB_SIZE)
        features["depth"] = tf.reshape(tf.decode_raw(features['depth'], tf.float32), self.DEPTH_SIZE)
        features["segmentation"] = tf.reshape(tf.decode_raw(features['segmentation'], tf.float32), self.BINARY_SIZE)
        features["skeleton"] = tf.reshape(tf.decode_raw(features['skeleton'], tf.float32), self.SKELETON_SIZE)
        features["skeleton_img"] = tf.reshape(tf.decode_raw(features['skeleton_img'], tf.float32), self.RGB_SIZE)
        features["length"] = features["length"][0]
        features["label"] = features["label"][0]
        features["id"] = features["id"][0]
        return features


if __name__ == '__main__':
    # Load a sample from tfrecords visualize RGB and Skeleton images. Note that we use Tensorflow eager execution here.
    # The dataset class works both in eager and static modes. However, the model and training codes only work in static
    # mode.
    import time
    import matplotlib.pyplot as plt
    from Skeleton import Skeleton
    tf.enable_eager_execution()

    mode = "validation"
    file_pattern = os.path.join("./../project1_skeleton/data", mode+"-??-of-??")

    #file_pattern = os.path.join("/cluster/project/infk/hilliges/lectures/mp19/project1", mode+"-??-of-??")
    print(file_pattern)

    dataset = TFRecordDataset(data_path=file_pattern,
                              batch_size=8,
                              normalize=False,
                              shuffle=False)

    data_iterator = dataset.get_iterator()
    '''
    batch = next(data_iterator)

    
    img_rgb = batch['rgb'][1][1]
    img_rgb = tf.cast(img_rgb,tf.int32)
    img_d = batch['depth'][1][1]
    img_s = batch['segmentation'][1][1]
    skeleton_img = batch['skeleton_img'][1][1]
    print(img_rgb)


    print(batch['rgb'].shape)
    print (batch['length'][1])
    print (batch['label'][1])
    print (batch['id'][1])



    plt.figure()
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(img_rgb)
    plt.subplot(2, 2, 2)
    plt.imshow(tf.squeeze(img_d), cmap='gray')
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(img_s)
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(skeleton_img)
    plt.axis("off")
    plt.show()
    '''

    # You can use this code snippet to read the entire dataset. It can be useful if you want to convert tfrecords
    # to numpy for custom preprocessing.
    
    num_samples = 0
    start_time = time.perf_counter()
    samples = []
    output_dir = "./newdata/" + mode 

    #output_dir = "/cluster/project/infk/courses/machine_perception_19/Shibainukawaii/pre_data/" + mode 
    n_shards = 20
    tfrecord_writers = n2t.create_tfrecord_writers(output_dir, n_shards)

    cnt = 0
    for batch in data_iterator:
        b_size = batch["rgb"].shape[0]
        num_samples += b_size

        for i in range(b_size):

            length = batch['length'][i]
            tmp = []

            print(length)

            for l in range(length):
                img_rgb = batch['rgb'][i][l]
                skeleton = Skeleton(batch['skeleton'][i][l])
                skeleton.resizePixelCoordinates()
                tmp.append(skeleton.toImage(img_rgb.shape[0], img_rgb.shape[1]))
        
            skeleton_img = np.array(tmp).astype(np.float32)

            print(skeleton_img.shape)
            cnt +=1
            print (cnt)

            sample = {
                "rgb"          : np.array(batch['rgb'][i][:length][:,:,:,::-1]),
                "depth"        : np.array(batch['depth'][i][:length]),
                "segmentation" : np.array(batch['segmentation'][i][:length]),
                "skeleton"     : np.array(batch['skeleton'][i][:length]),
                "skeleton_img" : skeleton_img,
                "length"      : np.array(batch['length'][i]),
                "label"       : np.array(batch['label'][i]),
                "id"          : np.array(batch['id'][i])
            }
            tfexample = n2t.to_tfexample(sample)
            n2t.write_tfexample(tfrecord_writers, tfexample)

    n2t.close_tfrecord_writers(tfrecord_writers)


    time_elapsed = (time.perf_counter() - start_time)
    print("Time elapsed {:.3f}".format(time_elapsed))
    print("# samples " + str(num_samples))
    


