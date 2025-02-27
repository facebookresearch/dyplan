# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SEED=${1:=1000}
DATASET=${2:="hotpotqa"}
OUTPUT_STR=${3:=""}
CONFIG_NAME=${4:="DRAGIN"}
echo "SEED: $SEED"
echo "DATASET: $DATASET"
echo "OUTPUT_STR: $OUTPUT_STR"
echo "CONFIG_NAME: $CONFIG_NAME"

echo "Running elastic search"
cd ./data/elasticsearch-7.17.9
nohup bin/elasticsearch &
cd ../../src
sleep 30

if [ $OUTPUT_STR != "" ]; then
    RESULT_DIR=../result/llama3_8b_chat_"$DATASET"/$OUTPUT_STR
fi

if [ $DATASET = "hotpotqa" ]; then
    DATASET="HotpotQA"
elif [ $DATASET = "2wikimultihopqa" ]; then
    DATASET="2WikiMultihopQA"
elif [ $DATASET = "musique" ]; then
    DATASET="Musique"
fi

if [ $OUTPUT_STR = "" ]; then
    python main.py -c ../config/Llama3-8b-instruct/"$DATASET"/"$CONFIG_NAME".json -d --seed $SEED
else
    python main.py -c ../config/Llama3-8b-instruct/"$DATASET"/"$CONFIG_NAME".json -d --seed $SEED -o $OUTPUT_STR
fi

if [ $OUTPUT_STR != "" ]; then
    python evaluate.py --dir $RESULT_DIR
fi
