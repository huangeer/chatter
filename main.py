from tensorflow.contrib.summary import summary
from tensorflow.python.ops import audio_ops as contrib_audio

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
