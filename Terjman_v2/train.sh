export CUDA_VISIBLE_DEVICES="2"

model_name="1B" # "3B", "1B", "Helsinki_240M", "Helsinki_77M_512", "Helsinki_77M_256"
max_len=512 # 256, 512, 1024

# python3 train.py
# accelerate launch train_v2.py \

# python3 train.py \
#     --model_name $model_name \
#     --max_len $max_len \

python3 train_any_to_any_lang.py \
    --model_name $model_name \
    --max_len $max_len \