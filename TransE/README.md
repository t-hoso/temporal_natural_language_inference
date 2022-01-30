# TransEs for Sentences

# setting
Download [atomic-2020 dataset](https://github.com/allenai/comet-atomic-2020)  
Then set ATOMIC_DIRECTORY where the dataset is

# run
set environment variables.  
set MODEL_NAME to {transe, transh, complex}  
set MAPPED_EMBEDDING_DIM, which is the dim of embedding used in models  
set TENSOR_BOARD_PATH and MODEL_SAVE_PATH wherever  
set LEARNING_RATE  
set EPOCHS  

then run  
```
python -m transe --model_name MODEL_NAME --mapped_embedding_dim MAPPED_EMBEDDING_DIM --relation isAfter --relation isBefore --tensorboard_path TENSOR_BOARD_PATH --lr LEARNING_RATE --n_epochs EPOCHS --model_save_path MODEL_SAVE_PATH 
```

Note that the option relation is multiple. See atomic-2020.