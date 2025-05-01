configs=(
    "h6_w7_c4_small_200.yaml"
    "h6_w7_c4_large.yaml"
    "h5_w5_c3_small.yaml"
    "h6_w7_c4_medium.yaml"
    "h7_w8_c5_small_200.yaml"
    "h7_w8_c5_small_400.yaml"
    "h7_w8_c5_small_600.yaml"
    "h9_w9_c5_small.yaml"
)

for config in "${configs[@]}"; do
    rm -rf datasets
    python train.py --config "configs/$config" --device cuda
done
