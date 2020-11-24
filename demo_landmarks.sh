LATENT_CODE_NUM=1
#rm -rf results/landmark
# Parameters for reference (attr_index, step_size, steps) (4: 0.005 400) (5: 0.01 100), (6: 0.1 200), (8 0.1 200)
CUDA_VISIBLE_DEVICES=0 python edit.py \
    -m stylegan_celebahq \
    -b boundaries/ \
    -n "$LATENT_CODE_NUM" \
    -o results/ \
    --task landmark \
    --method ours \
    --step_size 0.1 \
    --steps 200 \
    --attr_index 6\
    --condition\
    --direction 1 \
    --demo

