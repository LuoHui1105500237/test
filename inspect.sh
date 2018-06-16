#!/usr/bin/env sh
set -e

echo "import pb to tensorboard:................"
model_path=../freeze/quantized_graph.pb
#model_path=../freeze/frozen_graph.pb
log_path=../freeze/
python import_pb_to_tensorboard.py \
    --model_dir=${model_path} \
    --log_dir=${log_path}

echo "Done."