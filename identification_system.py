# ==============================================================================
# Constructor fot the architecture: BLSTM5
# UNAM IIMAS
# Authors: 	Ivette Velez
# 			Caleb Rascon
# 			Gibran Fuentes
#			Alejandro Maldonado
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os
import sys
import time
import glob
import json
import numpy as np
import soundfile as sf
import tensorflow as tf
import sounddevice as sd
import random
import librosa

from sklearn.metrics import roc_curve
from collections import namedtuple
from tensorflow.contrib import rnn
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using just one GPU in case of GPU 
# os.environ['CUDA_VISIBLE_DEVICES']= '0'

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
		print('no display found. Using non-interactive Agg backend')
		mpl.use('Agg')
import matplotlib.pyplot as plt

class Model():

	def __init__(self, 
		in_height,
		in_width,
		channels,
		label):

		self.in_height = in_height
		self.in_width = in_width
		self.channels = channels
		self.label = label

		# To avoid future errors initializing all the variables
		self.X1 = None
		self.X2 = None
		self.Y = None
		self.Y_pred = None
		self.label_pred = None
		self.label_true = None
		self.accuracy = None
		self.acc_batch = None
		self.loss = None
		self.Y_logt = None
		self.build()

	def build(self):
		""" Creates the model """
		self.def_input()
		self.def_params()
		self.def_model()
		self.def_output()
		self.def_loss()
		self.def_metrics()
		self.add_summaries()

	
	def def_input(self):
		""" Defines inputs """
		with tf.name_scope('input'):

			# Defining the entrance of the model
			self.X1 = tf.compat.v1.placeholder(tf.float32, [None, self.in_height, self.in_width, self.channels], name='X1')
			self.X2 = tf.compat.v1.placeholder(tf.float32, [None, self.in_height, self.in_width, self.channels], name='X2')
			self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.label], name='Y')

	def def_model(self):
		""" Defines the model """
		self.Y_logt = tf.constant(0, shape=[self.label]);		

	def def_output(self):
		""" Defines model output """
		with tf.name_scope('output'):
			self.Y_pred = tf.compat.v1.nn.softmax(self.Y_logt, name='Y_pred')
			self.label_pred = tf.math.argmax(self.Y_pred, 1, name='label_pred')
			self.label_true = tf.math.argmax(self.Y, 1, name='label_true')

	def def_metrics(self):
		""" Adds metrics """
		with tf.name_scope('metrics'):
			self.cmp_labels = tf.math.equal(self.label_true, self.label_pred)
			self.accuracy = tf.math.reduce_sum(tf.cast(self.cmp_labels, tf.float32), name='accuracy')
			self.acc_batch = tf.math.reduce_mean(tf.cast(self.cmp_labels, tf.float32))*100

	def add_summaries(self):
		""" Adds summaries for Tensorboard """
		with tf.name_scope('summaries'):
			tf.compat.v1.summary.scalar('loss', self.loss)
			tf.compat.v1.summary.scalar('accuracy', self.acc_batch)
			self.summary = tf.compat.v1.summary.merge_all()

	def def_loss(self):
		""" Defines loss function """
		self.loss = tf.constant(0)

	def def_params(self):
		""" Defines model parameters """		


class BLSTM5(Model):

	def __init__(self, 
		in_height,
		in_width,
		channels,
		label,
		num_hidden,
		layers):
		self.num_hidden = num_hidden
		self.layers = layers
		Model.__init__(self,in_height,in_width,channels,label)

	def def_loss(self):
		""" Defines loss function """
		with tf.name_scope('loss_1'):

			# cross entropy
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			self.loss = tf.math.reduce_mean(self.cross_entropy)

	def def_params(self):
		""" Defines model parameters """
		with tf.name_scope('params'):

			# For fully connected layer
			self.W_1 = tf.compat.v1.get_variable('weights1',[self.num_hidden * 2, self.label],initializer=tf.initializers.glorot_uniform())
			self.b_1 = tf.compat.v1.get_variable('bias1',[self.label],initializer=tf.initializers.glorot_uniform())


	def def_model(self):
		""" Defines the model """
		W_1 = self.W_1
		b_1 = self.b_1

		# Doing a reshape of the inputs
		r_X1 = tf.reshape(self.X1, [-1, self.in_height, self.in_width])
		r_X1 = tf.transpose(r_X1, (0,2,1))

		# BLSTM
		with tf.name_scope('blstm'):
			self.blstm_model  = self.blst_model(r_X1,reuse=False)
			
		# normalization
		with tf.name_scope('normalization'):					
			self.norm = tf.nn.l2_normalize(self.blstm_model, 1)

		# First fully connected layer
		with tf.name_scope('fc1'):		
			self.Y_logt = tf.matmul(self.norm, W_1) + b_1

	def lstm_cell(self,num_hidden,name,activation,reuse):
		return tf.nn.rnn_cell.LSTMCell(
			num_hidden, forget_bias= 1.0,
			initializer=tf.initializers.glorot_uniform(),
			activation=activation,
			name = name,
			reuse = reuse)

	def blst_model(self,input_,reuse=False):
		
		with tf.name_scope("model_lstm"):

			# For the dynamic BLSTM
			x = input_
			fw_lstm_cells_encoder = [self.lstm_cell(self.num_hidden,"fw_blstm_{}".format(i),tf.tanh,reuse) for i in range(self.layers )]
			bw_lstm_cells_encoder = [self.lstm_cell(self.num_hidden,"bw_blstm_{}".format(i),tf.tanh,reuse) for i in range(self.layers )]			
			(outputs, output_state_fw, output_state_bw) = rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_lstm_cells_encoder, cells_bw=bw_lstm_cells_encoder, inputs=x, dtype=tf.float32)
			res = outputs[:,-1,:]

		return res


class ModelBuilder():
	def __init__(self,		 
		in_height,
		in_width,
		channels,
		label):
		
		self.BLSTM5 = "BLSTM5"

		self.in_height = in_height
		self.in_width = in_width
		self.channels = channels
		self.label = label

		self.num_hidden = None
		self.layers = None
		self.BLSTM_flag = False

	def AddParamsBLSTM(self,num_hidden, layers):
		self.num_hidden = num_hidden
		self.layers = layers
		self.BLSTM_flag = True

	def BuildType(self,model_type):
		
		if model_type == self.BLSTM5:
			if not self.BLSTM_flag:
				return None

			model = BLSTM5(
				self.in_height,
				self.in_width,
				self.channels,
				self.label,
				self.num_hidden, 
				self.layers)

		return model


# ======================================================
# Functions and methods to generate the data
# ======================================================
def get_part(part,string):
	# Function that splits and string to get the desire part
	aux = string.split('/')
	a = aux[len(aux)-part-1]
	return a

def get_class(direction):
	# Getting the class of the audio
	class_index = direction.rfind('/')
	fixed_class = direction[class_index+1:len(direction)]

	class_index = fixed_class.rfind('_')
	if class_index >= 0:
		fixed_class = fixed_class[0:class_index]

	return fixed_class

# VAD functions
def pre_proccessing(audio, rate, pre_emphasis = 0.97, frame_size=0.02, frame_stride=0.01):
	emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
	frame_length, frame_step = frame_size * rate, frame_stride * rate	# Convert from seconds to samples
	audio_length = len(emphasized_audio) 
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(np.ceil(float(np.abs(audio_length - frame_length)) / frame_step))	# Make sure that we have at least 1 frame
	pad_audio_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_audio_length - audio_length))
	pad_audio = np.append(emphasized_audio, z) # Pad audio to make sure that all frames have equal number of samples without truncating any samples from the original audio
	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step)\
	, (frame_length, 1)).T
	frames = pad_audio[indices.astype(np.int32, copy=False)]
	return frames

def power_spect(audio, rate):
	frames = pre_proccessing(audio, rate)
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT_VAD))	# Magnitude of the FFT
	pow_frames = ((1.0 / NFFT_VAD) * ((mag_frames) ** 2))	# Power Spectrum
	return pow_frames

def mel_filter(audio, rate, nfilt = 40):
	pow_frames = power_spect(audio, rate)
	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))	# Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)	# Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))	# Convert Mel to Hz
	bin = np.floor((NFFT_VAD + 1) * hz_points / rate)
	fbank = np.zeros((nfilt, int(np.floor(NFFT_VAD / 2 + 1))))

	for m in range(1, nfilt + 1):
		 f_m_minus = int(bin[m - 1])	 # left
		 f_m = int(bin[m])						 # center
		 f_m_plus = int(bin[m + 1])		# right

		 for k in range(f_m_minus, f_m):
				fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
		 for k in range(f_m, f_m_plus):
				fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	

	filter_banks = np.dot(pow_frames, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)	# Numerical Stability
	return hz_points ,filter_banks

def voice_frecuency(audio,rate):
	frec_wanted = []
	hz_points, filter_banks = mel_filter(audio, rate)
	for i in range(len(hz_points)-2):
		 if hz_points[i]<= HIGHT_BAN and hz_points[i] >=LOW_BAN:
				frec_wanted.append(1)
		 else:
				frec_wanted.append(0)
	
	sum_voice_energy = np.dot(filter_banks, frec_wanted)/1e+6	## 1e+6 is use to reduce the audio amplitud 
	return(sum_voice_energy)

def get_points(aux, sr=16000, frame_size=0.02, frame_stride=0.01):
	flag_audio = False
	cont_silence = 0 
	init_audio = 0
	start =[]
	end = []
	min_frames = 40
	threshold = np.max(aux) * 0.04

	for i in range(len(aux)):
		if aux[i]	< threshold:
			cont_silence+=1

			if cont_silence == min_frames:
				if flag_audio == True:
					start.append(init_audio)
					end.append(i-min_frames+1)
					flag_audio = False
			
		if aux[i] > threshold:
			if flag_audio == False:
				init_audio = i
				flag_audio = True

			cont_silence=0

	if flag_audio == True:
		start.append(init_audio)
		end.append(len(aux))

	start = (np.array(start) * frame_stride * sr).astype(int)
	end = (np.array(end) * frame_stride * sr).astype(int)

	return start,end

def vad_analysis(audio, samplerate):
	# Analyzing the VAD of the audio
	voice_energy = voice_frecuency(audio, samplerate)
	start, end= get_points(voice_energy,samplerate)
	r_start = []
	r_end = []

	for i in range(0,start.shape[0]):
		if end[i] - start[i] > WINDOW:
			r_start.append(start[i])
			r_end.append(end[i])

	return np.array(r_start),np.array(r_end)


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

def generate_data(data_type, audio, start, end, samplerate = 16000):

	# Choosing randomly a window that fits the specifications
	option = random.randrange(0,len(start),1)
	index = random.randrange(start[option],end[option]-WINDOW,1)
	audio_data = audio[index:index+WINDOW]

	

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

	return data_audio, audio_data


def read_file(file):
	matrix = []
	for line in file:
		row = line.rstrip()
		matrix.append([row])
	return matrix

def load_db():

	database = []

	aux = glob.glob( os.path.join(DIRECTORY_DATABASE, '*.npy') )

	for i in range(0,len(aux)):
		row_class = get_class(aux[i])
		database.append([str(aux[i]), str(row_class)])

	return database

def create_data_array(data_audio):

	X1 = []
	X2 = []
	Y = []

	aux_X2 = np.zeros((IN_HEIGHT, IN_WIDTH, 1))
	Y_aux = np.zeros((LABEL)).tolist()

	X1.append(data_audio)
	X2.append(aux_X2)
	Y.append(Y_aux)

	return np.array(X1), np.array(X2), np.array(Y)


def statistics_audio(Y_Aux, total_ver_y_pred):

	unique_classes = np.unique(np.array(Y_Aux))
	class_value = np.zeros((len(unique_classes)))
	
	for number_class in range(0,len(unique_classes)):

		num_audios = 0

		for row_pred in range(0,len(total_ver_y_pred)):

			if Y_Aux[row_pred] == unique_classes[number_class]:
				class_value[number_class] = class_value[number_class] + total_ver_y_pred[row_pred]
				num_audios+=1

		class_value[number_class] = class_value[number_class]/num_audios
		print(unique_classes[number_class], ': ', class_value[number_class])

	# Initializing the value of the class
	value_class = 'unknown'
	value_y_pred = 0

	# Choosing only the class with the highest score above 0.5
	for row_pred in range(0,len(class_value)):
	
		if class_value[row_pred] > model_threshold and class_value[row_pred] > value_y_pred:

			value_class = unique_classes[row_pred]
			value_y_pred = class_value[row_pred]	

	return value_class


def get_number_audio(direction):
	# Getting the class of the audio
	class_index = direction.rfind('/')
	fixed_class = direction[class_index+1:len(direction)]

	class_index = fixed_class.rfind('.')
	fixed_class = fixed_class[0:class_index]

	class_index = fixed_class.rfind('_')
	if class_index >= 0:
		number_audio = fixed_class[class_index+1:]

	return int(number_audio)

def save_data(database, data, name, audio_data):

	number_audio = 0

	for i in range(0,len(database)):
		class_data = database[i][1]
		numb_audio = get_number_audio(database[i][0])

		if class_data == name and numb_audio > number_audio:
			number_audio = numb_audio

	number_audio+=1
	name_speaker = name + '_' + str(number_audio)
	np.save(DIRECTORY_DATABASE+'/'+str(name_speaker), data)
	sf.write(DIRECTORY_DATABASE+'/'+name_speaker+'.wav', audio_data, SAMPLERATE)





# ======================================================
# Loading the configuration for the model
# ======================================================
configuration_file = str(sys.argv[1])
if configuration_file == "":
	print("ERROR: you need to define param: config_model_datatype.json ")
	exit(0)

PARAMS = None

with open(configuration_file, 'r') as f:
	f = f.read()
	PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

# Doing the process required for the loaded data
DIRECTORY_DATABASE = PARAMS.PATHS.database
DIRECTORY_WEIGHTS = PARAMS.PATHS.path_weights

SECONDS = PARAMS.DATA_GENERATOR.window
SAMPLERATE = PARAMS.DATA_GENERATOR.sample_rate
WINDOW = int(float(SECONDS*SAMPLERATE))
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

ARCHITECTURE = PARAMS.TRAINING.architecture
CHANNELS= PARAMS.TRAINING.channels
LABEL = PARAMS.TRAINING.label
RESTORE_WEIGHTS = PARAMS.TRAINING.restore_weights
NUM_HIDDEN = PARAMS.TRAINING.num_hidden
LAYERS = PARAMS.TRAINING.layers

# Variables for VAD analysis
NFFT_VAD = 512
LOW_BAN = 300
HIGHT_BAN = 3000

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

model_threshold = 0

if os.path.exists(DIRECTORY_DATABASE) == False:
	os.mkdir(DIRECTORY_DATABASE)

# ======================================================
# Creating the network model
# ======================================================
builder = ModelBuilder(in_height = IN_HEIGHT,
		in_width = IN_WIDTH,
		channels = CHANNELS,
		label = LABEL)

# If BLSTM load extra params
if ARCHITECTURE == "BLSTM5":
	builder.AddParamsBLSTM(NUM_HIDDEN, LAYERS)

network = builder.BuildType(ARCHITECTURE)

if network == None:
	print("Bad configuration parameters in model.")
	exit(0)


# ======================================================
# Running the model
# ======================================================

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# opens session
with tf.Session() as sess:
	
	# initialize variables (params)
	sess.run(init_op)
	
	saver = tf.compat.v1.train.Saver()
	if RESTORE_WEIGHTS == "True":
		print("Restoring weights")
		saver.restore(sess, DIRECTORY_WEIGHTS)
	elif RESTORE_WEIGHTS == "False":
		print("No weights restored")
		print("You need to restore the corresponding weights to continue.")
		exit(0)
	else:
		print("ERROR: you need to indicate if the weights should or shouldn't be restored.")
		exit(0)


	while True:

		os.system('clear') 
		print('Please press any character when you want to start speaking') 
		option = input()
		audio_length = WINDOW + SAMPLERATE * 1
		recording = np.zeros(int(audio_length))
		recording = sd.rec(int(audio_length), samplerate=SAMPLERATE, channels=1)
		time.sleep(audio_length/SAMPLERATE) 
		print('Thanks, you can stop speaking')	

		# If you want to try offline data, load here, in recording the audio you want to verify

		start,end = vad_analysis(recording, SAMPLERATE)

		if len(start)>0:	   	

			database = load_db()
			row_matrix = 0
			pred = []
			total_class = []

			data_audio, audio_data= generate_data(DATA_TYPE, recording, start, end, SAMPLERATE)
			X1, X2, Y = create_data_array(data_audio)

			feed_dict = {network.X1: X1, network.X2: X2, network.Y : Y}
			fetches = [network.norm]
			X1_model = sess.run(fetches, feed_dict=feed_dict)

			for row_matrix in range(0,len(database)):

				aux_name = database[row_matrix][1]
				X2_model = np.load(database[row_matrix][0])

				score = np.sum(X1_model * X2_model)
				total_class.append(aux_name)
				pred.append(score)

			name_speaker = statistics_audio(total_class, pred)

			print("\nYou are", name_speaker, "\n" )

			speaker_correct = 'n'
			if name_speaker != 'unknown':
				print("Are you this speaker? (y--> yes, n-->no)")
				speaker_correct = input()

			if speaker_correct == 'n':
				print('Please write your name: ')
				name_speaker = input()			

			save_data(database, X1_model, name_speaker, audio_data)

		else:
			print("\nPlease speak louder and enough time.")
			input("\nPress any character to continue")
