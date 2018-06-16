#!/usr/bin/env sh
set -e

echo "print:............."
checkpoints=./checkpoints/vgg_16.ckpt
all=False
tensor=vgg_16/fc8/weights
python inspect_checkpoint.py \
    --file_name=${checkpoints} \
    --all_tensors=${all} \
    --tensor_name=${tensor}
echo "Done."