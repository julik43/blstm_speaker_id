# blstm_speaker_id
In this repository an approach to speaker identification through verification using a BLSTM network is presented.

There are three posibilities with the code given:
1. Train your own models.
2. Built an Identification system with the weights of your trained model.
3. Use an identification system with weights given.


Notes:

-For compatibility with https://github.com/julik43/Online-Identification-of-New-Speakers the model even when only needs one input has two.

-To avoid compatibility issues with the version of TensorFlow, the conda environment used during this exploration is available as "environment.yml".


# 1. Train your own models

Update the correct path on the folder data for train_speakers.txt, valid_speakers.txt and test_speakers.txt to the files you are going to use.

Use a configuration like one in the configurations folder and run the models like this:

bash run.sh configurations/config_BLSTM5_256_hidden_5000_loc_specdb_adam_30_epochs_0.5s.json 0 1 tf_1_14_gpu

This run.sh script recieves the desired configuration of the model, the GPU to be used (in case you only have 1 is always 0), the amount of processes to generate data and the conda environment to be used. The basic case for the processes is 1, it is recommended with the configurations given to use as processes 1, 2, or 4.

For this project, VoxCeleb database was changed from m4a format to flac format using the bash code "change_m4a_to_flac.sh".

Note: it is important to mention that "change_m4a_to_flac.sh" localize the audios of the third level of folders from the path given.

Note 2: In the folder configurations you can find the rest of configurations used during this project.

Note 3: For this project a VAD was applied to all audios of each section to be used, this information was stored in numpy files for its start and end (in this repository is available in the folder data), and is mandatory to have one of this. If you don't want to use any VAD for training, please create this file and use as start 0 and as end the length of the audio. This VAD must match each line with the ones in train_speakers.txt, valid_speakers.txt and test_speakers.txt respectively.
In this repository the VAD used is available in the folder data. However if you want to create this files you can run:

python find_VAD.py list_of_audios.txt length

where list_of_audios.txt is the list of audios you want to find the VAD, and length is the amount of samples you want to find with voice activity in the audio, for example 16000 (1 second with samplerate of 16000).


# 2. Built an Identification system with the weights of your trained model.

Update the path of the weights in identification_system.json and all data needed.

Run the identification model like this:

python identification_system.py identification_system.json


# 3. Use an identification system with weights given.

Download the complete repository. This will have in the folder "SpecdB_0.5" the weights for the model of SpecdB of 0.5s.

Run the identification model like this:

python identification_system.py identification_system.json


