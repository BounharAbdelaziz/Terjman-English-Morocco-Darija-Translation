export CUDA_VISIBLE_DEVICES="1"

model_name="3B" # "3B", "1B", "Helsinki_240M", "Helsinki_77M_512", "Helsinki_77M_256"
max_len=1024 # 256, 512, 1024
train_nllb=1 # 0, 1

# python3 train.py
# accelerate launch train_v2.py \
python3 train.py \
    --model_name $model_name \
    --max_len $max_len \
    --train_nllb $train_nllb