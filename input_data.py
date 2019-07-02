from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.summary import summary
from tensorflow.python.ops import audio_ops as contrib_audio ####用的是旧版的脚本

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

####silence and unknown_word label and index
MAX_NUM_WAVS_PER_CLASS = 2**27-1
SILENCE_LABEL='_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED=59185
def prepare_words_list(wanted_words):
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
def which_set(filename, validation_percentage, testing_percentage):
  base_name = os.path.basename(filename)
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) *(100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else: 
    result = 'training'
  return result
def load_wav_file(filename):
  wav_loader = tf.io.read_file(filename)
  wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
  return tf.keras.layers.Flatten()(wav_decoder.audio)
def save_wav_file(filename, wav_data, sample_rate):
  wav_encoder = tf.audio.encode_wav(np.reshape(wav_data, (-1, 1)),sample_rate)
  wav_saver = tf.io.write_file(filename, wav_encoder)
def get_features_range(model_settings):
  if model_settings['preprocess'] == 'average':
    features_min = 0.0
    features_max = 127.5
  elif model_settings['preprocess'] == 'mfcc':
    features_min = -247.0
    features_max = 30.0
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                    ' "average")' % (model_settings['preprocess']))
  return features_min, features_max

class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""
  ####(1)
  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings, summaries_dir):
    if data_dir:
      self.data_dir = data_dir
      self.maybe_download_and_extract_dataset(data_url, data_dir)
      self.prepare_data_index(silence_percentage, unknown_percentage, 
                              wanted_words, validation_percentage,
                              testing_percentage)
      self.prepare_background_data()
    self.execute_above(data_url,data_dir,silence_percentage, unknown_percentage, 
                              wanted_words, validation_percentage,
                              testing_percentage)
  ####(2)
  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    self.wav_filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, self.wav_filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (self.wav_filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        logging.error('Failed to download URL: %s to folder: %s', data_url,
                         filepath)
        logging.error('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      logging.info('Successfully downloaded %s (%d bytes)', self.wav_filename,
                      statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  ####(3)
  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
  ####(4)
  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data

    search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                               '*.wav')
    for wav_path in gfile.Glob(search_path):
      wav_loader = tf.io.read_file(wav_path)
      wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
      wav_data =tf.keras.layers.Flatten()(wav_decoder.audio)
      self.background_data.append(wav_data)
    if not self.background_data:
      raise Exception('No background wav files were found in ' + search_path)
  
  ####(5)
  def execute_above(self,data_url,data_dir,silence_percentage, unknown_percentage, 
                              wanted_words, validation_percentage,
                              testing_percentage):
    self.maybe_download_and_extract_dataset(data_url, data_dir)
    self.prepare_data_index(silence_percentage, unknown_percentage, 
                              wanted_words, validation_percentage,
                              testing_percentage)
    prepare_background_data()
    
  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])
  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, summaries_dir,mode):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the transformed samples, and list of label indexes

    Raises:
      ValueError: If background samples are too short.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      # If we're time shifting, set up the offset for this sample.
      if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
      else:
        time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]
      input_dict = {
          'wav_filename_': sample['file'],
          'time_shift_padding_': time_shift_padding,
          'time_shift_offset_': time_shift_offset,
      }
      # Choose a section of background noise to mix in.
      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= model_settings['desired_samples']:
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (model_settings['desired_samples'], len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = tf.reshape(background_clipped,[desired_samples, 1])
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0
      input_dict['background_data_'] = background_reshaped
      input_dict['background_volume_'] = background_volume
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        input_dict['foreground_volume_'] = 0
      else:
        input_dict['foreground_volume_'] = 1
      # Run the graph to produce the output audio.
      
      wav_loader = tf.io.read_file(input_dict['wav_filename_'])
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      # Allow the audio sample's volume to be adjusted.
      scaled_foreground = tf.multiply(
          wav_decoder.audio,input_dict['foreground_volume_'])
      # Shift the sample's start position, and pad any gaps with zeros.
      padded_foreground = tf.pad(
          scaled_foreground,input_dict['time_shift_padding_'],mode='CONSTANT',
          constant_values=0,name=None)
      sliced_foreground = tf.slice(padded_foreground,
                                   input_dict['time_shift_offset_'],
                                   [desired_samples, -1])
      
      # Mix in background noise.

      background_mul = tf.multiply(input_dict['background_data_'],
                                   input_dict['background_volume_'])
      
      sliced_foreground = tf.dtypes.cast(sliced_foreground, tf.dtypes.float64)
      background_mul = tf.dtypes.cast(background_mul, tf.dtypes.float64)
      background_add = tf.add(background_mul, sliced_foreground)
      
      background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
      background_clamp = tf.dtypes.cast(background_clamp, dtype=tf.dtypes.float32)
      # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
      spectrogram = contrib_audio.audio_spectrogram(
          background_clamp,
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True)
      tf.summary.image(
          'spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
      # The number of buckets in each FFT row in the spectrogram will depend on
      # how many input samples there are in each window. This can be quite
      # large, with a 160 sample window producing 127 buckets for example. We
      # don't need this level of detail for classification, so we often want to
      # shrink them down to produce a smaller result. That's what this section
      # implements. One method is to use average pooling to merge adjacent
      # buckets, but a more sophisticated approach is to apply the MFCC
      # algorithm to shrink the representation.
      if model_settings['preprocess'] == 'average':
        self.output_ = tf.nn.pool(
            tf.expand_dims(spectrogram, -1),
            window_shape=[1, model_settings['average_window_width']],
            strides=[1, model_settings['average_window_width']],
            pooling_type='AVG',
            padding='SAME')
        tf.summary.image('shrunk_spectrogram', self.output_, max_outputs=1)
      elif model_settings['preprocess'] == 'mfcc':
        self.output_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['fingerprint_width'])
        tf.summary.image(
            'mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
      else:
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
                         ' "average")' % (model_settings['preprocess']))

      # Merge all the summaries and write them out to /tmp/retrain_logs (by
      # default)
      self.merged_summaries_ = tf.summary.create_file_writer('tmp/retrain_dir')
      if summaries_dir:
        self.summary_writer_ = tf.summary.create_file_writer(summaries_dir + '/data')
      
  
      data_tensor = self.output_
      data_tensor = tf.dtypes.cast(data_tensor,dtype=tf.dtypes.float64)
      data[i - offset, :] =tf.compat.v1.layers.flatten(
          data_tensor,
          name=None,
          data_format='channels_last')
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
    return data, labels
  ####(8)
  def get_features_for_wav(self, wav_filename, model_settings):
    
    """Applies the feature transformation process to the input_wav.

    Runs the feature generation process (generally producing a spectrogram from
    the input samples) on the WAV file. This can be useful for testing and
    verifying implementations being run on other platforms.

    Args:
      wav_filename: The path to the input audio file.
      model_settings: Information about the current model being trained.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      Numpy data array containing the generated features.
    """
    desired_samples = model_settings['desired_samples']
    input_dict = {
        'wav_filename_': self.wav_filename,
        'time_shift_padding_': [[0, 0], [0, 0]],
        'time_shift_offset_': [0, 0],
        'background_data_': np.zeros([desired_samples, 1]),
        'background_volume_': 0,
        'foreground_volume_': 1,
    }
    # Run the graph to produce the output audio.
    data_tensor = self.output_(input_dict)
    return data_tensor
  ####(9)
  def get_unprocessed_data(self, how_many, model_settings, mode):
    """Retrieve sample data for the given partition, with no transformations.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    candidates = self.data_index[mode]
    
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    
    desired_samples = model_settings['desired_samples']
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = [] 
    for i in range(sample_count):
      if how_many == -1:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      input_dict = {'wav_filename_': sample['file']}
      if sample['label'] == SILENCE_LABEL:
        input_dict['foreground_volume_'] = 0
      else:
        input_dict['foreground_volume_'] = 1
      wav_loader = tf.io.read_file(input_dict['wav_filename_'])
      wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
    
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                    input_dict['foreground_volume_'])
      data[i, :] = scaled_foreground.flatten()
      label_index = self.word_to_index[sample['label']]
      labels.append(words_list[label_index])
    return data, labels
    
