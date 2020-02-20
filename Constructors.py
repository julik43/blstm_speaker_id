# ==============================================================================
# Running with tensorflow 1.14 and above including 2.0
# Constructor fot the architectures: 
#			* BLSTM
# UNAM IIMAS
# Authors: 	Ivette Velez
#			Alejandro Maldonado
# ==============================================================================
# Notes:
# tf.contrib.layers.xavier_initializer change to tf.initializers.glorot_uniform
# tf.contrib.rnn.LSTMCell change to tf.nn.rnn_cell.LSTMCell
# tf.contrib.layers.l2_regularizer to tf.keras.regularizers.l2
# For tensorflow 2.0: tf.compat.v1.nn.rnn_cell.DropoutWrapper will become tf.nn.RNNCellDropoutWrapper


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
# import keras as K
from sklearn.metrics import roc_curve
from collections import namedtuple
from tensorflow.contrib import rnn
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# from tensorflow.contrib import layers

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

visible_devices = str(sys.argv[2])
if visible_devices == "":
	print("ERROR: you need to define the visible_devices")
	exit(0)

# Using just one GPU in case of GPU 
os.environ['CUDA_VISIBLE_DEVICES']= visible_devices

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
		print('no display found. Using non-interactive Agg backend')
		mpl.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K

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


class TFRecordReader():
	def __init__(self,
		train_tf_records_files = None,
		validation_tf_records_files = None,
		test_tf_records_files = None,
		parse_function = None,
		batch_size = None):

		self.train_tf_records_files = train_tf_records_files
		self.validation_tf_records_files = validation_tf_records_files
		self.test_tf_records_files = test_tf_records_files
		self.parse_function = parse_function
		self.batch_size = batch_size

		# ====================================
		# Training Dataset
		# ====================================
		train_dataset = tf.data.TFRecordDataset(self.train_tf_records_files)
		
		# Parse the record into tensors.
		train_dataset = train_dataset.map(self.parse_function)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_dataset = train_dataset.batch(self.batch_size)

		# ====================================
		# Validation Dataset
		# ====================================
		validation_dataset = tf.data.TFRecordDataset(self.validation_tf_records_files)

		# Parse the record into tensors.
		validation_dataset = validation_dataset.map(self.parse_function)
		validation_dataset = validation_dataset.shuffle(buffer_size=10000)
		validation_dataset = validation_dataset.batch(self.batch_size)

		# ====================================
		# test Dataset
		# ====================================
		test_dataset = tf.data.TFRecordDataset(self.test_tf_records_files)

		# Parse the record into tensors.
		test_dataset = test_dataset.map(self.parse_function)
		test_dataset = test_dataset.batch(self.batch_size)

		# ====================================
		# Defining the handler
		# ====================================
		self.train_handle = tf.compat.v1.placeholder(tf.string, shape=[])
		self.validation_handle = tf.compat.v1.placeholder(tf.string, shape=[])
		self.test_handle = tf.compat.v1.placeholder(tf.string, shape=[])
		
		self.train_iterator = tf.compat.v1.data.Iterator.from_string_handle(self.train_handle, train_dataset.output_types, train_dataset.output_shapes)
		self.validation_iterator = tf.compat.v1.data.Iterator.from_string_handle(self.validation_handle, validation_dataset.output_types, validation_dataset.output_shapes)
		self.test_iterator = tf.compat.v1.data.Iterator.from_string_handle(self.test_handle, test_dataset.output_types, test_dataset.output_shapes)

		self.train_next_element = self.train_iterator.get_next()
		self.validation_next_element = self.validation_iterator.get_next()
		self.test_next_element = self.test_iterator.get_next()
		
		# Defining the iterators
		self.training_iterator = train_dataset.make_initializable_iterator()
		self.validation_iterator = validation_dataset.make_initializable_iterator()
		self.test_iterator = test_dataset.make_initializable_iterator()

class Parser():
	def __init__(self,
		in_height,
		in_width,
		label,
		):

		self.in_height = in_height
		self.in_width = in_width
		self.label = label

	def parse_function(self,proto):

		# Defining the features to be loaded from the tfrecords file
		features = tf.io.parse_single_example(proto,
				# Defaults are not specified since both keys are required.
				features={
						'audio1': tf.io.FixedLenFeature([], tf.string),
						'audio2': tf.io.FixedLenFeature([], tf.string),
						'label': tf.io.FixedLenFeature([], tf.string),
				})

		# Convert from a scalar string tensor to a float32 tensor
		audio1= tf.decode_raw(features['audio1'], tf.float32)
		audio1 = tf.reshape(audio1,(self.in_height,self.in_width,1)) # If we want a flat vector #image.set_shape([in_heigth_size*in_with_size])
		audio1 = tf.cast(audio1, tf.float32)

		# Loading the second image as the first was loaded
		audio2= tf.decode_raw(features['audio2'], tf.float32)
		audio2 = tf.reshape(audio2,(self.in_height,self.in_width,1))
		audio2 = tf.cast(audio2, tf.float32)

		# Loading the labels 
		label= tf.decode_raw(features['label'], tf.int64)
		label.set_shape([self.label])

		return audio1, audio2, label

def file_observer(path, num_files):
	# Adquiring the data for the database
	database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))

	while database.shape[0] < num_files:
		print('Waiting for files')
		time.sleep(1)
		database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))

	return


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
DIRECTORY_TRAIN = PARAMS.PATHS.directory_train
DIRECTORY_VALID = PARAMS.PATHS.directory_valid
DIRECTORY_TEST = PARAMS.PATHS.directory_test
DIRECTORY_WEIGHTS = PARAMS.PATHS.path_weights

N_ROW_TF_RECORD = PARAMS.DATA_GENERATOR.n_row_tf_record
ROUNDS_TRAIN = PARAMS.DATA_GENERATOR.rounds_train
ROUNDS_VALID = PARAMS.DATA_GENERATOR.rounds_valid
ROUNDS_TEST = PARAMS.DATA_GENERATOR.rounds_test
NUM_TRAIN_FILES = PARAMS.DATA_GENERATOR.num_train_files
NUM_VALID_FILES = PARAMS.DATA_GENERATOR.num_valid_files
NUM_TEST_FILES = PARAMS.DATA_GENERATOR.num_test_files

SECONDS = PARAMS.DATA_GENERATOR.window
WINDOW = float(SECONDS*PARAMS.DATA_GENERATOR.sample_rate)
MS = 1.0/PARAMS.DATA_GENERATOR.sample_rate
NPERSEG = int(PARAMS.DATA_GENERATOR.nperseg/MS)
NOVERLAP = int(PARAMS.DATA_GENERATOR.noverlap/MS)
NFFT = PARAMS.DATA_GENERATOR.nfft
N_USED = PARAMS.DATA_GENERATOR.n_used
DATA_TYPE = PARAMS.DATA_GENERATOR.data_type
SIZE_TIME = int((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP))+1
RE_READ = PARAMS.DATA_GENERATOR.re_read

if DATA_TYPE == "MFCC":
	SIZE_TIME = SIZE_TIME+2

IN_WIDTH = SIZE_TIME
IN_HEIGHT = N_USED

BATCH_SIZE = PARAMS.TRAINING.batch_size
NUM_EPOCHS = PARAMS.TRAINING.num_epochs
LEARNING_RATE = PARAMS.TRAINING.learning_rate
TYPE_OPT = PARAMS.TRAINING.type_opt
ARCHITECTURE = PARAMS.TRAINING.architecture
CHANNELS= PARAMS.TRAINING.channels
LABEL = PARAMS.TRAINING.label
RESTORE_WEIGHTS = PARAMS.TRAINING.restore_weights

NUM_HIDDEN = PARAMS.TRAINING.num_hidden
LAYERS = PARAMS.TRAINING.layers
MOMENTUM = PARAMS.TRAINING.momentum

RESNET_CONFIG = []

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

NAME_OUTPUT = ARCHITECTURE + '_' + DATA_TYPE + '_' + str(SECONDS) + 's' + '_' + TYPE_OPT + '_' + str(LEARNING_RATE) + '_' + str(NUM_EPOCHS) + '_' + str(NFFT) + '_' + str(N_USED)

if ARCHITECTURE == 'BLSTM5':
	NAME_OUTPUT = NAME_OUTPUT + '_hidden_'+ str(NUM_HIDDEN) + '_layers_'+str(LAYERS)

OUTPUT_FILE = open(NAME_OUTPUT+'_results.txt', 'w')
OUTPUT_FOLDER = NAME_OUTPUT

if os.path.exists(OUTPUT_FOLDER) == False:
	os.mkdir(OUTPUT_FOLDER)

DELETE_FILE_EVERY = int(N_ROW_TF_RECORD/BATCH_SIZE)

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

if TYPE_OPT == "grad":
	optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(network.loss)

elif TYPE_OPT == "RMSProp":
	optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum = MOMENTUM).minimize(network.loss)

elif TYPE_OPT == "ADAM":
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(network.loss)

# ======================================================
# Creating the parser 
# ======================================================
parser = Parser(in_height = IN_HEIGHT,
		in_width = IN_WIDTH,
		label = LABEL)

# ======================================================
# Creating the tf records reader
# ======================================================
train_file = []
validation_file = []
test_file = []

for i in range(0,NUM_TRAIN_FILES):
	train_file.append(DIRECTORY_TRAIN+'/'+ 'train_' +str(i) + '.tfrecords')

for i in range(0,NUM_VALID_FILES):
	validation_file.append(DIRECTORY_VALID+'/'+ 'validation_' +str(i) + '.tfrecords')

for i in range(0,NUM_TEST_FILES):
	test_file.append(DIRECTORY_TEST+'/'+ 'test_' +str(i) + '.tfrecords')

tf_records_reader = TFRecordReader(train_tf_records_files = train_file,
		validation_tf_records_files = validation_file,
		test_tf_records_files = test_file,
		parse_function = parser.parse_function,
		batch_size = BATCH_SIZE)

# ======================================================
# Running the model
# ======================================================

# The op for initializing the variables.
init_op = tf.group(tf.compat.v1.global_variables_initializer(),tf.compat.v1.local_variables_initializer())

# opens session
with tf.compat.v1.Session() as sess:
	
	# writers for TensorBorad
	train_writer = tf.compat.v1.summary.FileWriter('graphs/train_'+ NAME_OUTPUT)
	valid_writer = tf.compat.v1.summary.FileWriter('graphs/valid_'+ NAME_OUTPUT)
	test_writer = tf.compat.v1.summary.FileWriter('graphs/test_'+ NAME_OUTPUT)
	train_writer.add_graph(sess.graph)

	extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

	# initialize variables (params)
	sess.run(init_op)
	training_handle = sess.run(tf_records_reader.training_iterator.string_handle())
	validation_handle = sess.run(tf_records_reader.validation_iterator.string_handle())
	test_handle = sess.run(tf_records_reader.test_iterator.string_handle())

	saver = tf.compat.v1.train.Saver()
	if RESTORE_WEIGHTS == "True":
		print("Restoring weights")
		saver.restore(sess, DIRECTORY_WEIGHTS)
	elif RESTORE_WEIGHTS == "False":
		print("No weights restored")
	else:
		print("ERROR: you need to indicate if the weights should or shouldn't be restored.")

	# Initializing the step for train and validation
	step_train = 1
	step_valid = 1
	step_test = 1

	acc_train = 0
	acc_valid = 0
	acc_test = 0

	for n_epochs in range(NUM_EPOCHS):

		print('Epoch: ', n_epochs)
		init_time = time.time()
		reread = 0
		
		for n_rounds_train in range(ROUNDS_TRAIN*RE_READ):			

			# print('Round train: ', n_rounds_train)

			if reread == RE_READ:
				reread = 0

			n_file_train = 1
			file_observer(DIRECTORY_TRAIN, NUM_TRAIN_FILES)
			sess.run(tf_records_reader.training_iterator.initializer)

			# Running the training
			while True:

				try:
					# Training with train data
					X1_array, X2_array, Y_array = sess.run(tf_records_reader.train_next_element, feed_dict={tf_records_reader.train_handle: training_handle})
					feed_dict = {network.X1: X1_array, network.X2: X2_array, network.Y : Y_array}
					fetches = [optimizer, network.loss, network.accuracy, network.summary, network.Y_pred, network.label_true, network.label_pred, extra_update_ops ]
					_,train_loss, train_acc, train_summary, y_pred, l_true, l_pred,_ = sess.run(fetches, feed_dict=feed_dict)
					train_writer.add_summary(train_summary, step_train)

					acc_train = acc_train + train_acc

					# Printing the results every 50 batch
					if step_train % 50 == 0:
						msg = "I{:3d} loss_train: ({:6.8f}), acc_train(batch, global): ({:6.8f},{:6.8f})"
						msg = msg.format(step_train, train_loss, train_acc/BATCH_SIZE, acc_train/(BATCH_SIZE*step_train))
						print(msg)
						OUTPUT_FILE.write(msg + '\n')

					if (n_file_train)%DELETE_FILE_EVERY == 0 and reread == (RE_READ-1):
						number_file = int((n_file_train/DELETE_FILE_EVERY)-1)
						os.remove(train_file[number_file])

					step_train += 1		 
					n_file_train+=1

				except tf.errors.OutOfRangeError:
					# If the data ended the while must be broken
					break;

			reread = reread+1

		print('End training epoch')
		# Saving the weightsin every epoch
		save_path = saver.save(sess, str(OUTPUT_FOLDER+'/'+ str(n_epochs) +'weights.ckpt'))

		print("Time in minutes for the " + str(n_epochs)+ " epoch: " + str((time.time() - init_time)/60))

		for n_rounds_valid in range(ROUNDS_VALID):
			
			n_file_valid = 1
			file_observer(DIRECTORY_VALID, NUM_VALID_FILES)
			sess.run(tf_records_reader.validation_iterator.initializer)

			# Running the validation
			while True:

				try:
					# evaluation with valid data
					X1_array, X2_array, Y_array = sess.run(tf_records_reader.validation_next_element, feed_dict={tf_records_reader.validation_handle: validation_handle})
					feed_dict = {network.X1: X1_array, network.X2: X2_array, network.Y : Y_array}
					fetches = [network.loss, network.accuracy, network.summary ]
					valid_loss, valid_acc, valid_summary = sess.run(fetches, feed_dict=feed_dict)
					valid_writer.add_summary(valid_summary, step_train)

					acc_valid = acc_valid + valid_acc

					# Printing the results every 100 batch
					if step_valid % 100 == 0:
						msg = "I{:3d} loss_valid: ({:6.8f}), acc_valid(batch, global): ({:6.8f},{:6.8f})"
						msg = msg.format(step_valid, valid_loss, valid_acc/BATCH_SIZE, acc_valid/(BATCH_SIZE*step_valid))
						print(msg)
						OUTPUT_FILE.write(msg + '\n')

					if (n_file_valid)%DELETE_FILE_EVERY == 0:
						number_file = int((n_file_valid/DELETE_FILE_EVERY)-1)
						os.remove(validation_file[number_file])

					step_valid += 1		 
					n_file_valid+=1

				except tf.errors.OutOfRangeError:
					# If the data ended the while must be broken
					break;


	for n_rounds_test in range(ROUNDS_TEST):
			
		n_file_test = 1
		file_observer(DIRECTORY_TEST, NUM_TEST_FILES)
		sess.run(tf_records_reader.test_iterator.initializer)

		# Running the test
		while True:

			try:
				# evaluation with test data
				X1_array, X2_array, Y_array = sess.run(tf_records_reader.test_next_element, feed_dict={tf_records_reader.test_handle: test_handle})
				feed_dict = {network.X1: X1_array, network.X2: X2_array, network.Y : Y_array}
				fetches = [network.loss, network.accuracy, network.summary, network.Y_pred, network.label_pred, network.label_true, network.cmp_labels]
				test_loss, test_acc, test_summary, Y_pred, l_pred, l_true, cmp_labels = sess.run(fetches, feed_dict=feed_dict)
				test_writer.add_summary(test_summary, step_train)

				acc_test = acc_test + test_acc

				# Printing the results every 100 batch
				# if True:
				if step_test % 100 == 0:
					msg = "I{:3d} loss_test: ({:6.8f}), acc_test(batch, global): ({:6.8f},{:6.8f})"
					msg = msg.format(step_test, test_loss, test_acc/BATCH_SIZE, acc_test/(BATCH_SIZE*step_test))
					print(msg)
					OUTPUT_FILE.write(msg + '\n')

				if (n_file_test)%DELETE_FILE_EVERY == 0:
					number_file = int((n_file_test/DELETE_FILE_EVERY)-1)
					os.remove(test_file[number_file])

				step_test += 1		 
				n_file_test+=1

			except tf.errors.OutOfRangeError:
				# If the data ended the while must be broken
				break;