#!/usr/bin/env sh
set -e
echo "freeze:............."
input=Placeholder
output=ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax_5/Reshape_1,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/softmax/Reshape_1

graph_path=../freeze/inference.pb
checkpoint_path=../test_log_finetune/model.ckpt-5000
freeze_path=../freeze/frozen_graph.pb

python freeze_graph.py \
    --input_graph=${graph_path} \
    --input_checkpoint=${checkpoint_path} \
    --output_graph=${freeze_path} \
    --output_node_names=${output}

echo "optimize:................"
optima_name=../freeze/optimized_graph.pb
python optimize_for_inference.py \
    --input=${freeze_path} \
    --output=${optima_name} \
    --frozen_graph=True \
    --input_names=${input} \
    --output_names=${output}

echo "import pb to tensorboard:................"
log_path=../freeze/
python import_pb_to_tensorboard.py \
    --model_dir=${optima_name} \
    --log_dir=${log_path}

echo "Done."