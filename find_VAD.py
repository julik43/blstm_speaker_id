# ==============================================================================
# Find the VAD for a list of audios
# UNAM IIMAS
# Authors: 	Ivette Velez
# ==============================================================================

from __future__ import print_function

import sys
import numpy as np
import soundfile as sf

from scipy import signal
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

# ======================================================
# Functions and methods for VAD
# ======================================================
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

def read_file(file):
	matrix = []
	for line in file:
		row = line.rstrip()
		matrix.append(row)
	return matrix


# ======================================================
# Loading the configuration for the model
# ======================================================
if len(sys.argv) < 2:
    print("ERROR: you need to give a file with the path to each audio.")
    exit(0)

if sys.argv[1] == "":
    print("ERROR: you need to give a file with the paths to each audio.")
    exit(0)

audio_file = str(sys.argv[1])
    
if len(sys.argv) < 3:
    print("WARNING: you did not give any WINDOW size for the VAD search. 8000 samples (0.5s) is used as default.")
    WINDOW = 8000
else:
	WINDOW = int(sys.argv[2])

# Variables for VAD analysis
NFFT_VAD = 512
LOW_BAN = 300
HIGHT_BAN = 3000

input_file	= open(audio_file,'r')
list_files = read_file(input_file)
matrix_vad_start = []
matrix_vad_end = []

for n_file in range(0, len(list_files)):

	name = list_files[n_file]
	print(n_file, name)
	audio,samplerate = sf.read(name)
	start,end = vad_analysis(audio, samplerate)

	matrix_vad_start.append(start)
	matrix_vad_end.append(end)

np.save("VAD_start", matrix_vad_start)
np.save("VAD_end", matrix_vad_end)