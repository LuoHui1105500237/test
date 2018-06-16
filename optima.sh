#!/usr/bin/env sh
set -e

echo "optimize:................"
model_path=../freeze/frozen_graph.pb
output_path=../freeze/optimized_graph.pb
input=Placeholder
output=ssd_300_vgg/block11_box/Reshape,ssd_300_vgg/softmax_5/Reshape_1,ssd_300_vgg/block10_box/Reshape,ssd_300_vgg/softmax_4/Reshape_1,ssd_300_vgg/block9_box/Reshape,ssd_300_vgg/softmax_3/Reshape_1,ssd_300_vgg/block8_box/Reshape,ssd_300_vgg/softmax_2/Reshape_1,ssd_300_vgg/block7_box/Reshape,ssd_300_vgg/softmax_1/Reshape_1,ssd_300_vgg/block4_box/Reshape,ssd_300_vgg/softmax/Reshape_1
python optimize_for_inference.py \
    --input=${model_path} \
    --output=${output_path} \
    --frozen_graph=True \
    --input_names=${input} \
    --output_names=${output}

echo "Done."