LATENT_CODE_NUM=1
# rm -rf results/attribute
# Note that to edit age and eyeglasses on StyleGAN, larger step size, say 0.3, can get better results.
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
    --condition\
    --demo

