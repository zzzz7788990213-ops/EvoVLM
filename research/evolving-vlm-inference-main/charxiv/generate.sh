model_name=YOUR_MODEL # custom name for model
split=val # choose from [val, test]
mode=reasoning # choose from [reasoning, descriptive]
model_path=YOUR_CKPT # path to the model weights
model_api=YOUR_KEY # API key IFF testing proprietary models

### generate response for open-weight models ###
python src/generate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --model_path $model_path 

### generate response for proprietary models ###
# python src/generate.py \
#     --model_name $model_name \
#     --split $split \
#     --mode $mode \
#     --model_path $model_path \
#     --model_api $model_api
