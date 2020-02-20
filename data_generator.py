# ==============================================================================
# Constructor fot the architectures: V7, VGG11, VGG13, VGG16, ResNet 50
# UNAM IIMAS
# Authors: 	Ivette Velez
# 			Caleb Rascon
# 			Gibran Fuentes
#			Alejandro Maldonado
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.signal import lfilter, hamming
from scipy.fftpack import fft

import os.path
import os
import sys
import glob
import json
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import time
import librosa

from scipy import signal
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
from soundfile import SoundFile

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======================================================
# Functions and methods need it
# ======================================================
def get_part(part,string):
	# Function that splits and string to get the desire part
	aux = string.split('/')
	a = aux[len(aux)-part-1]
	return a

def get_class(direction):
	# Getting the class of the audio
	fixed_class = get_part(2,direction)
	return fixed_class

def verify_length_vad(start, end, length):

	r_start = []
	r_end = []
	
	for i in range(0,start.shape[0]):
		if end[i] - start[i] > WINDOW:
			r_start.append(start[i])
			r_end.append(end[i])

	if len(r_start) == 0 and length>WINDOW:
		r_start = [0]
		r_end = [length]

	return np.array(r_start), np.array(r_end)

# Functions to generate the data 
def to_dB_mag(magnitude,MIN_AMP,AMP_FAC):
    magnitude = np.maximum(magnitude, np.max(magnitude) / float(MIN_AMP))
    magnitude = 20. * np.log10(magnitude * AMP_FAC)
    return magnitude

def preemp(audio, p):
	"""Pre-emphasis filter."""
	return lfilter([1., -p], 1, audio)

def segment_axis( a, length, overlap=0, axis=None, end='cut', endvalue=0):
		"""Generate a new array that chops the given array along the given axis
		into overlapping frames.
		example:
		>>> segment_axis(arange(10), 4, 2)
		array([[0, 1, 2, 3],
			   [2, 3, 4, 5],
			   [4, 5, 6, 7],
			   [6, 7, 8, 9]])
		arguments:
		a	   The array to segment
		length  The length of each frame
		overlap The number of array elements by which the frames should overlap
		axis	The axis to operate on; if None, act on the flattened array
		end	 What to do with the last frame, if the array is not evenly
				divisible into pieces. Options are:
				'cut'   Simply discard the extra values
				'wrap'  Copy values from the beginning of the array
				'pad'   Pad with a constant value
		endvalue	The value to use for end='pad'
		The array is not copied unless necessary (either because it is unevenly
		strided and being flattened or because end is set to 'pad' or 'wrap').
		"""

		if axis is None:
			a = np.ravel(a) # may copy
			axis = 0

		l = a.shape[axis]

		if overlap >= length:
			# raise ValueError, "frames cannot overlap by more than 100%"
			print("frames cannot overlap by more than 100%")
		if overlap < 0 or length <= 0:
			# raise ValueError, "overlap must be nonnegative and length must "\
			# 				  "be positive"
			print("overlap must be nonnegative and length must be positive")

		if l < length or (l-length) % (length-overlap):
			if l>length:
				roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
				rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
			else:
				roundup = length
				rounddown = 0
			assert rounddown < l < roundup
			assert roundup == rounddown + (length-overlap) \
				   or (roundup == length and rounddown == 0)
			a = a.swapaxes(-1,axis)

			if end == 'cut':
				a = a[..., :rounddown]
			elif end in ['pad','wrap']: # copying will be necessary
				s = list(a.shape)
				s[-1] = roundup
				b = np.empty(s,dtype=a.dtype)
				b[..., :l] = a
				if end == 'pad':
					b[..., l:] = endvalue
				elif end == 'wrap':
					b[..., l:] = a[..., :roundup-l]
				a = b

			a = a.swapaxes(-1,axis)


		l = a.shape[axis]
		# print("axis",axis)
		# print("l",l)
		# print("a",a)
		if l == 0:
			# raise ValueError, \
			# 	  "Not enough data points to segment array in 'cut' mode; "\
			# 	  "try 'pad' or 'wrap'"

			print("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")
		assert l >= length
		assert (l-length) % (length-overlap) == 0
		n = 1 + (l-length) // (length-overlap)
		s = a.strides[axis]
		newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
		newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

		try:
			return np.ndarray.__new__(np.ndarray, strides=newstrides,
									  shape=newshape, buffer=a, dtype=a.dtype)
		except TypeError:
			warnings.warn("Problem with ndarray creation forces copy.")
			a = a.copy()
			# Shape doesn't change but strides does
			newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
						 + a.strides[axis+1:]
			return np.ndarray.__new__(np.ndarray, strides=newstrides,
									  shape=newshape, buffer=a, dtype=a.dtype)	

def get_emph_spec(audio, nperseg=256, noverlap = 96, nfft=512, fs=16000):
	# Function to generate the emphasized spectrogram
	prefac = 0.97
	w = hamming(nperseg, sym=0)
	extract = preemp(audio, prefac)
	framed = segment_axis(extract, nperseg, noverlap) * w
	spec = np.abs(fft(framed, nfft, axis=-1))
	return spec

def generate_data(data_type, audio, samplerate = 16000):

	audio_data = audio

	if data_type == 'Spec':
		f, t, Sxx = signal.spectrogram(audio_data, fs = samplerate,	window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
		Hxx = StandardScaler().fit_transform(Sxx)
		data_audio = np.reshape(Hxx[0:IN_HEIGHT,:],(IN_HEIGHT,IN_WIDTH,1))

	elif data_type == 'SpecMag':
		f, t, Sxx = signal.spectrogram(audio_data, fs = samplerate,	window=('hann'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode = 'magnitude')
		Hxx = StandardScaler().fit_transform(Sxx)
		data_audio = np.reshape(Hxx[0:IN_HEIGHT,:],(IN_HEIGHT,IN_WIDTH,1))

	elif data_type == 'SpecdB':
		f, t, Sxx = signal.spectrogram(audio_data, fs = samplerate,	window=('hann'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1, mode='magnitude')
		Sxx = to_dB_mag(Sxx, 10000, 10000)
		Hxx = (Sxx-Sxx.mean())/Sxx.std()
		data_audio = np.reshape(Hxx[0:IN_HEIGHT,:],(IN_HEIGHT,IN_WIDTH,1))
	
	elif data_type == 'EmphSpec':
		spec = get_emph_spec(audio_data, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,fs=samplerate)
		spec = np.transpose(spec, (1,0))
		data_audio = np.reshape(spec[0:IN_HEIGHT,0:IN_WIDTH],(IN_HEIGHT, IN_WIDTH,1))	

	elif data_type == 'EmphSpecdB':
		spec = get_emph_spec(audio_data, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,fs=samplerate)
		spec = np.transpose(spec, (1,0))
		Sxx = to_dB_mag(spec, 10000, 10000)
		Hxx = (Sxx-Sxx.mean())/Sxx.std()
		data_audio = np.reshape(Hxx[0:IN_HEIGHT,0:IN_WIDTH],(IN_HEIGHT, IN_WIDTH,1))	

	elif data_type == 'MFCC':
		spec = librosa.feature.mfcc(y=audio_data, sr=samplerate, n_mfcc=IN_HEIGHT, n_fft=NFFT, hop_length=NPERSEG-NOVERLAP, win_length=NPERSEG)
		data_audio = np.reshape(spec[0:IN_HEIGHT,0:IN_WIDTH],(IN_HEIGHT, IN_WIDTH,1))	

	return data_audio

# Functions to write the data in TFRecord format
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(audio1, audio2, labels, name):
	"""Converts a dataset to tfrecords."""
	num_examples = audio1.shape[0]
	rows = audio1.shape[1]
	cols = audio1.shape[2]
	depth = audio1.shape[3]
	filename = os.path.join(name + '.tfrecords')

	print('Writing', filename)
	writer = tf.python_io.TFRecordWriter(filename)
	for index in range(num_examples):

		audio1_raw = audio1[index].tostring()
		audio2_raw = audio2[index].tostring()
		label = labels[index].tostring()
		
		example = tf.train.Example(features=tf.train.Features(feature={
				'height': _int64_feature(rows),
				'width': _int64_feature(cols),
				'depth': _int64_feature(depth),
				'audio1': _bytes_feature(audio1_raw),
				'audio2': _bytes_feature(audio2_raw),
				'label': _bytes_feature(label)}))
		writer.write(example.SerializeToString())
	writer.close()

def extra_params(section, step_tf):

	data_start = []
	data_end = []

	if section == 0:
		path = DIRECTORY_TRAIN
		name_base = (path+'/train')
		total_files = int(NUM_TRAIN_FILES*ROUNDS_TRAIN/step_tf)
		input_file	= open(FILE_TRAIN,'r')
		num_total_files_per_round = NUM_TRAIN_FILES
		if DATA_VAD_TRAIN_START != "":
			data_start = np.load(DATA_VAD_TRAIN_START, encoding='bytes', allow_pickle=True)
			data_end = np.load(DATA_VAD_TRAIN_END, encoding='bytes', allow_pickle=True)

	elif section == 1:			
		path = DIRECTORY_VALID
		name_base = (path+'/validation')
		total_files = int(NUM_VALID_FILES*ROUNDS_VALID/step_tf)
		input_file	= open(FILE_VALID,'r')
		num_total_files_per_round = NUM_VALID_FILES
		if DATA_VAD_VALID_START != "":
			data_start = np.load(DATA_VAD_VALID_START, encoding='bytes', allow_pickle=True)
			data_end = np.load(DATA_VAD_VALID_END, encoding='bytes', allow_pickle=True)

	elif section == 2 :
		path = DIRECTORY_TEST
		name_base = (path+'/test')
		total_files = int(NUM_TEST_FILES*ROUNDS_TEST/step_tf)
		input_file	= open(FILE_TEST,'r')
		num_total_files_per_round = NUM_TEST_FILES
		if DATA_VAD_TEST_START != "":
			data_start = np.load(DATA_VAD_TEST_START, encoding='bytes', allow_pickle=True)
			data_end = np.load(DATA_VAD_TEST_END, encoding='bytes', allow_pickle=True)

	return path, name_base, total_files, input_file, num_total_files_per_round, data_start, data_end	

def read_file(file):
	matrix = []
	for line in file:
		row = line.rstrip()
		class_row = str(get_class(row))
		matrix.append([row, class_row])
	return matrix

def select_classes(matrix, data_start, data_end):

	matrix = np.array(matrix)
	unique_classes = np.unique(matrix[:, 1])

	# Selecting only the classes needed
	# unique_classes = unique_classes[0:LABEL]
	unique_classes = unique_classes[len(unique_classes)-LABEL:len(unique_classes)]

	aux_matrix = []
	aux_data_start = []
	aux_data_end = []

	for i in range(0,len(matrix)):
		if len(np.where(unique_classes == matrix[i][1])[0]):
			aux_matrix.append([matrix[i][0], matrix[i][1]])
			aux_data_start.append(data_start[i])
			aux_data_end.append(data_end[i])

	return aux_matrix, unique_classes, aux_data_start, aux_data_end
	
def read_audio_segment(file, pos, length):
	myfile = SoundFile(file)
	myfile.seek(pos)
	audio = myfile.read(length)
	myfile.close()
	return audio

def choose_audio(aux):

	while True:
		index_list = int(len(aux)*random.random())
		chosen_audio = aux[index_list][0]
		start = aux[index_list][1]
		end = aux[index_list][2]

		if len(start) > 0:			
			option = int(len(start)*random.random())
			factor = end[option]-WINDOW - start[option]
			audio_index = int(factor*random.random()) + start[option] 
			try:
				audio = read_audio_segment(chosen_audio, audio_index, WINDOW)
				return audio
			except Exception as e:
				print('Error with audio: ', chosen_audio)

def classification_data_array(dictionary, classes):

	X1 = []
	X2 = []
	Y = []
	rows = len(matrix)

	aux_X2 = np.zeros((IN_HEIGHT, IN_WIDTH, 1))
	total_classes = len(classes)

	while len(X1)< N_ROW_TF_RECORD:

		# print(len(X1))

		#=============================================
		# Two audios of the same class
		#=============================================
		id_class = int(total_classes*random.random())
		aux = dictionary[classes[id_class]]
		audio_1 = choose_audio(aux)
		data_audio_1= generate_data(DATA_TYPE, audio_1, SAMPLERATE)

		X1.append(data_audio_1)
		X2.append(aux_X2)

		Y_aux = np.zeros((LABEL)).tolist()
		Y_aux[id_class] = 1
		Y.append(Y_aux)

	return X1, X2, Y

def create_dict(matrix, data_start, data_end):

	dictionary = {}

	for i in range(0,len(matrix)):
		
		if len(data_end[i])>0:
			length = data_end[i][len(data_end[i])-1]
		else:
			length = 0

		aux_start, aux_end = verify_length_vad(data_start[i], data_end[i], length)

		if len(aux_start) > 0:
			if (matrix[i][1] in dictionary) ==  False:
				dictionary[matrix[i][1]] = []

			aux = dictionary[matrix[i][1]]
			aux.append([matrix[i][0], aux_start, aux_end])

	return dictionary

# ======================================================
# Loading the configuration for the model
# ======================================================
configuration_file = str(sys.argv[1])
if configuration_file == "":
    print("ERROR: you need to define param: config_model_datatype.json ")
    exit(0)
    
initial_number_tf = int(sys.argv[2])
step_tf = int(sys.argv[3])

if initial_number_tf == "":
    print("ERROR: you need to define the initial number of the TFRecord file (recommend: python data_generator 0 1)")
    exit(0)

if step_tf == "":
    print("ERROR: you need to define the step for the TFRecord file (recommend: python data_generator 0 1)")
    exit(0)

PARAMS = None

with open(configuration_file, 'r') as f:
    f = f.read()
    PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

# Doing the process required for the loaded data
DIRECTORY_TRAIN = PARAMS.PATHS.directory_train
DIRECTORY_VALID = PARAMS.PATHS.directory_valid
DIRECTORY_TEST = PARAMS.PATHS.directory_test
FILE_TRAIN = PARAMS.PATHS.file_train
FILE_VALID = PARAMS.PATHS.file_valid
FILE_TEST = PARAMS.PATHS.file_test
DATA_VAD_TRAIN_START = PARAMS.PATHS.data_vad_train_start
DATA_VAD_TRAIN_END = PARAMS.PATHS.data_vad_train_end
DATA_VAD_VALID_START = PARAMS.PATHS.data_vad_valid_start
DATA_VAD_VALID_END = PARAMS.PATHS.data_vad_valid_end
DATA_VAD_TEST_START = PARAMS.PATHS.data_vad_test_start
DATA_VAD_TEST_END = PARAMS.PATHS.data_vad_test_end

N_ROW_TF_RECORD = PARAMS.DATA_GENERATOR.n_row_tf_record
ROUNDS_TRAIN = PARAMS.DATA_GENERATOR.rounds_train
ROUNDS_VALID = PARAMS.DATA_GENERATOR.rounds_valid
ROUNDS_TEST = PARAMS.DATA_GENERATOR.rounds_test
NUM_TRAIN_FILES = PARAMS.DATA_GENERATOR.num_train_files
NUM_VALID_FILES = PARAMS.DATA_GENERATOR.num_valid_files
NUM_TEST_FILES = PARAMS.DATA_GENERATOR.num_test_files

SECONDS = PARAMS.DATA_GENERATOR.window
SAMPLERATE = PARAMS.DATA_GENERATOR.sample_rate
WINDOW = int(SECONDS*SAMPLERATE)
MS = 1.0/SAMPLERATE
NPERSEG = int(PARAMS.DATA_GENERATOR.nperseg/MS)
NOVERLAP = int(PARAMS.DATA_GENERATOR.noverlap/MS)
NFFT = PARAMS.DATA_GENERATOR.nfft
N_USED = PARAMS.DATA_GENERATOR.n_used
DATA_TYPE = PARAMS.DATA_GENERATOR.data_type
SIZE_TIME = int((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP))+1

if DATA_TYPE == "MFCC":
	SIZE_TIME = SIZE_TIME+2

IN_WIDTH = SIZE_TIME
IN_HEIGHT = N_USED

NUM_EPOCHS = PARAMS.TRAINING.num_epochs
LABEL = PARAMS.TRAINING.label

# ======================================================
# Creating the data generator
# ======================================================
if os.path.exists(str(DIRECTORY_TRAIN)) == False:
	os.mkdir(str(DIRECTORY_TRAIN))

if os.path.exists(str(DIRECTORY_VALID)) == False:
	os.mkdir(str(DIRECTORY_VALID))

if os.path.exists(str(DIRECTORY_TEST)) == False:
	os.mkdir(str(DIRECTORY_TEST))

# Compute for the desired number of epochs.
for n_epochs in range(NUM_EPOCHS):

	limit = 3 if n_epochs == (NUM_EPOCHS-1) else 2
	init = 0

	for section in range(init,limit):

		path, name_base, total_files, input_file, num_total_files_per_round, data_start, data_end = extra_params(section, step_tf)		
		matrix = read_file(input_file)

		# if label is greater than 2 is because the db is for classification
		# We will use X1 for the data and X2 will only have zeros matrixes
		# Y will have a one hot coding of the class
		matrix, classes_selected, data_start, data_end = select_classes(matrix, data_start, data_end)

		dictionary = create_dict(matrix, data_start, data_end)
		classes = list(dictionary.keys())
		classes.sort()

		row_matrix = initial_number_tf
		num_files = initial_number_tf

		for n_file in range(total_files):			
			X1, X2, Y = classification_data_array(dictionary, classes)
			name = name_base + '_'+str(num_files)

			X1_array = np.array(X1, dtype = np.float32)
			X2_array = np.array(X2, dtype = np.float32)
			Y_array = np.array(Y, dtype=np.int64)

			# Veryfing that the data to write is not in use
			database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))
			exists = np.where(database==(name + '.tfrecords'))

			while len(exists[0]) > 0:
				database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))
				exists = np.where(database==(name + '.tfrecords'))

			convert_to(X1_array,X2_array,Y_array,name)

			num_files += step_tf							

			if num_files >= num_total_files_per_round:
				num_files = initial_number_tf