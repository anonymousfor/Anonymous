LATENT_CODE_NUM=1
#rm -rf results/attribute
CUDA_VISIBLE_DEVICES=0 python edit.py \
    -m pggan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/ \
    --task attribute \
    --method ours \
    --step_size 0.2 \
    --steps 40 \
    --attr_index 0\
    --demo

