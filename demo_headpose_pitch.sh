LATENT_CODE_NUM=1
#rm -rf results/head_pose
# direction controls if the method increases or decreases logit values
CUDA_VISIBLE_DEVICES=0 python edit.py \
    -m stylegan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/ \
    --task head_pose \
    --method ours \
    --step_size 0.01 \
    --steps 2000 \
    --attr_index 1\
    --condition\
    --direction -1 \
    --demo

