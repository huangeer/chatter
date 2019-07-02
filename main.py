from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.summary import summary
from tensorflow.python.ops import audio_ops as contrib_audio

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
FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectogram timeslices.',)
parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_dir',
      help='Where to save summary logs for TensorBoard.')
parser.add_argument(
      '--wanted_words',
      type=str,
      default='backward,bed,bird,cat,dog,down,eight,four,five,follow',
      help='Words to use (others will be added to an unknown label)',)
parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc" or "average"')

FLAGS, unparsed = parser.parse_known_args()

logging.set_verbosity(logging.INFO)
model_settings = prepare_model_settings(
      len(prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)

audio_processor = AudioProcessor(
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
    
    logits, dropout_prob = create_model(
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
    predicted_indices = tf.math.argmax(logits, 1)
    correct_prediction = tf.math.equal(predicted_indices, ground_truth_input)

    confusion_matrix = tf.math.confusion_matrix(
                              ground_truth_input, predicted_indices, num_classes=label_count)

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
        with tf.name_scope('cross_entropy'):
          cross_entropy_mean = tf.keras.losses.sparse_categorical_crossentropy(
                                                ground_truth_input, logits)
        predicted_indices = tf.math.argmax(logits, 1)
        predicted_indices = tf.dtypes.cast(predicted_indices, dtype=tf.dtypes.int64)
        ground_truth_input = tf.dtypes.cast(ground_truth_input, dtype=tf.dtypes.int64)
        correct_prediction = tf.math.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.math.confusion_matrix(
                                   ground_truth_input, predicted_indices, num_classes=label_count)
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
