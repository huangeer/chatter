from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.summary import summary
from tensorflow.python.ops import audio_ops as contrib_audio
from tensorflow.examples.speech_command import model
from tensorflow.examples.speech_command import input_data

import hashlib
import math
import os
import random
import re
import sys 
import tarfile 
import numpy as np 

from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from absl import logging

import functools
import argparse


data_url='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
data_dir='/tmp/speech_dataset/'
background_volume=0.1
background_frequency=0.8
silence_percentage=10.0
unknown_percentage=10.0
time_shift_ms=100.0
testing_percentage=10    
validation_percentage=10      
sample_rate=16000    
clip_duration_ms=1000      
window_size_ms=30.0
window_stride_ms=10.0
feature_bin_count=40
how_many_training_steps='15000,3000'
eval_step_interval=400
learning_rate='0.001,0.0001'
batch_size=100
summaries_dir='/tmp/retrain_dir'
wanted_words='backward,bed,bird,cat,dog,down,eight,four,five,follow'    
train_dir='/tmp/speech_commands_train'
save_step_interval=100
start_checkpoint=''
model_architecture='conv'
check_nans=False
quantize=False
preprocess='mfcc'
     



logging.set_verbosity(logging.INFO)
model_settings = model.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)

audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir)

fingerprint_size = model_settings['fingerprint_size']
label_count = model_settings['label_count']
time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
if len(training_steps_list) != len(learning_rates_list):
  raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

if FLAGS.quantize:
  fingerprint_min, fingerprint_max = get_features_range(
        model_settings)
  fingerprint_input = tf.quantization.fake_quant_with_min_max_args(
        fingerprint_size, fingerprint_min, fingerprint_max)
else:
  fingerprint_input = fingerprint_size

start_step = 1

control_dependencies = []
if FLAGS.check_nans:
  checks = tf.compat.v1.add_check_numerics_ops()
  control_dependencies = [checks]

training_steps_max = np.sum(training_steps_list)
for training_step in xrange(start_step, training_steps_max + 1):
  # Figure out what the current learning rate is.
  training_steps_sum = 0
  for i in range(len(training_steps_list)):
    training_steps_sum += training_steps_list[i]
    if training_step <= training_steps_sum:
      learning_rate_value = learning_rates_list[i]
      break
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, FLAGS.summaries_dir,'training')
    fingerprint_input=train_fingerprints  
    ground_truth_input=train_ground_truth
    learning_rate_input=learning_rate_value
    dropout_prob=0.5
    
    logits, dropout_prob = model.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)
    ground_truth_input_tensor=tf.Variable(tf.convert_to_tensor(ground_truth_input))
    logits_tensor = tf.Variable(logits)
    def cross_entropy_mean(x,y):
      return tf.keras.losses.sparse_categorical_crossentropy(x, y)

    with tf.name_scope('train'):
      c_without_any_args = functools.partial(cross_entropy_mean, x=ground_truth_input_tensor, y=logits_tensor) 
      train_step = tf.keras.optimizers.SGD(lr=learning_rate_input, momentum=0.0, 
                                           decay=0.0, nesterov=False).minimize(c_without_any_args,logits_tensor)
    ground_truth_input_tensor=tf.reshape(ground_truth_input_tensor,[100,1])
    predicted_indices = tf.math.argmax(logits, 1)
    correct_prediction = tf.math.equal(predicted_indices, ground_truth_input)

    confusion_matrix = tf.math.confusion_matrix(
                              ground_truth_input_tensor, predicted_indices, num_classes=label_count)

    evaluation_step = tf.math.reduce_mean(tf.dtypes.cast(correct_prediction, tf.float64))

    with tf.name_scope('eval'):
      tf.summary.record_if(cross_entropy_mean(ground_truth_input_tensor,logits_tensor))
      tf.summary.record_if(evaluation_step)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    increment_global_step = tf.Variable(global_step, global_step + 1)
    merged_summaries = tf.summary.create_file_writer(FLAGS.summaries_dir + '/eval')
    with merged_summaries.as_default():
        tf.summary.flush()
    train_writer = tf.summary.create_file_writer(FLAGS.summaries_dir + '/train')
    with train_writer.as_default():
        tf.summary.flush()
    validation_writer = tf.summary.create_file_writer(FLAGS.summaries_dir + '/validation')
    with validation_writer.as_default():
        tf.summary.flush()
     

    if FLAGS.start_checkpoint:
      models.load_variables_from_checkpoint(FLAGS.start_checkpoint)
      start_step = global_step.eval()##

    logging.info('Training from step: %d ', start_step)

    with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
      f.write('\n'.join(audio_processor.words_list))

    train_summary=merged_summaries
    train_accuracy=evaluation_step
    cross_entropy_value=cross_entropy_mean(ground_truth_input_tensor,logits_tensor)
    cross_entropy_value=tf.reduce_sum(cross_entropy_value)/100

    
    logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)

    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, FLAGS.summaries_dir,'validation'))
        fingerprint_input = validation_fingerprints,
        ground_truth_input = validation_ground_truth,
        dropout_prob=1.0
        ground_truth_input_tensor=tf.Variable(tf.convert_to_tensor(ground_truth_input))
        logits, dropout_prob = model.create_model(
                               fingerprint_input,
                               model_settings,
                               FLAGS.model_architecture,
                               is_training=True)
        logits_tensor=tf.Variable(tf.convert_to_tensor(logits))
        with tf.name_scope('cross_entropy'):
          cross_entropy_mean = tf.keras.losses.sparse_categorical_crossentropy(
                                                ground_truth_input_tensor, logits_tensor)
        predicted_indices = tf.math.argmax(logits, 1)
        predicted_indices = tf.dtypes.cast(predicted_indices, dtype=tf.dtypes.int64)
        ground_truth_input = tf.dtypes.cast(ground_truth_input, dtype=tf.dtypes.int64)
        ground_truth_input_tensor=tf.reshape(ground_truth_input_tensor,[predicted_indices.shape[0],])
        correct_prediction = tf.math.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.math.confusion_matrix(
                                   ground_truth_input_tensor, predicted_indices, num_classes=label_count)
        evaluation_step = tf.math.reduce_mean(tf.dtypes.cast(correct_prediction, tf.float32))
        with tf.name_scope('eval'):
          tf.summary.histogram('cross_entropy', cross_entropy_mean)
          tf.summary.histogram('accuracy', evaluation_step)
        merged_summaries = tf.summary.create_file_writer(FLAGS.summaries_dir + '/eval')
        with merged_summaries.as_default():
            tf.summary.flush()
        validation_summary=merged_summaries
        validation_accuracy=evaluation_step
        conf_matrix = confusion_matrix
        
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

  if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
set_size = audio_processor.set_size('testing')
logging.info('set_size=%d', set_size)
total_accuracy = 0
total_conf_matrix = None  
for i in xrange(0, set_size, FLAGS.batch_size):
  test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing')  
  fingerprint_input=test_fingerprints
  ground_truth_input=test_ground_truth
  dropout_prob=1.0
  predicted_indices = tf.math.argmax(logits, 1)
  correct_prediction = tf.math.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.math.confusion_matrix(
                             ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.math.reduce_mean(tf.dtypes.cast(correct_prediction, tf.float32))
  test_accuracy=evaluation_step,
  conf_matrix = confusion_matrix

  batch_size = min(FLAGS.batch_size, set_size - i)
  total_accuracy += (test_accuracy * batch_size) / set_size
  if total_conf_matrix is None:
    total_conf_matrix = conf_matrix
  else:
    total_conf_matrix += conf_matrix
logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))

logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,set_size))
