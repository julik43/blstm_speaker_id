#!/bin/bash
configFile=$1
visibleDevice=$2
numProcc=$3
condaEnv=$4

if [ -z $configFile ]; then
	echo "There is not configuration file"
	echo "Goodbye."
elif [ -z $visibleDevice ]; then
	echo "There is not visible devices"
	echo "Goodbye."
elif [ -z $numProcc ]; then
	echo "There is not number of processes to generate data. Basic case is 1"
	echo "Goodbye."
elif [ -z $condaEnv ]; then
	echo "There is not conda environment"
	echo "Goodbye."
else
	model="model_"$visibleDevice
	echo "Creating tmux "$model
	tmux new -d -s $model
	tmux send-keys -t $model.0 "conda activate $condaEnv" ENTER
	tmux send-keys -t $model.0 "python Constructors.py $configFile $visibleDevice" ENTER

	for (( n=0; n<$numProcc; n++ ))
	do
		data_gen="data_generator_"$n"_"$visibleDevice
		echo "Creating tmux "$data_gen
		tmux new -d -s $data_gen
		tmux send-keys -t $data_gen.0 "conda activate $condaEnv" ENTER
		tmux send-keys -t $data_gen.0 "python data_generator.py $configFile $n $numProcc" ENTER
		sleep 2
	done
fi
