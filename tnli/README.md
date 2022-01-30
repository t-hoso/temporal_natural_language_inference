# Temporal Natural Language Inference
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Study over newly created dataset

# Setting
Download our dataset

Download snli, mnli

Install Self-Explaining Model by adding `__init__.py` files.  
Please make sure that the versions of some required libraries are different, but use the version of ours.
Then install TranseE for sentence.  

then

```
cd tnli
```

# Preprocess

## Environment Variables
DATA_DIR: where datasets are in

place folds dir in it with split dataset

SNLI_TEXT_DIR: dirname under DATA_DIR

place snli raw text data in it

MNLI_TEXT_DIR: dirname under DATA_DIR

SNLI_TRAIN_FILENAME

SNLI_DEV_FILENAME

SNLI_TEST_FILENAME

MNLI_TRAIN_FILENAME

MNLI_DEV_MATCHED_FILENAME

MNLI_DEV_MISMATCHED_FILENAME

TRANSE_PRETRAINED_FILENAME

this is one of todos, the names should be one.

## preprocess
`python -m preprocess kind_of_preprocess`  

you need to run  
```
python -m preprocess sentence_transformer  
```

## train  
```
cd training_test  
python -m train --dataset_type sbert_fold \
    --fold 1
    --batch_size 16 \
    --num_warmup_steps 10000 \
    --epochs 40 \
    --lr 5e-5 \
    --eps 1e-6 \
    --betas 0.9 --betas 0.999 \
    --weight_decay 0.01 \
    --tensorboard_path logs
    --lamb 1.0
    --loss_fn_name cross_entropy_loss
    --optimizer_name adam
    --model_name ffn
    --model_save_path model_data
```

