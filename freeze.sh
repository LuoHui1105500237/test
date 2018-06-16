#!/usr/bin/env sh
set -e

echo "freeze:............."
graph_path=../freeze/inference.pb
checkpoint_path=../log_finetune/model.ckpt-5000
output_path=../freeze/frozen_graph.pb
output=ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax_5/Reshape_1,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/softmax/Reshape_1
python freeze_graph.py \
    --input_graph=${graph_path} \
    --input_checkpoint=${checkpoint_path} \
    --output_graph=${output_path} \
    --output_node_names=${output}

echo "Done."