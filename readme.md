# Simple Template for PyTorch

## Modules
Image denoise is used as base task. The template including following modules
1) dataset
2) network(model)
3) utils
4) test code
5) train code

## How to run?
### test
```
python test.py --test_root $TEST_DATA_ROOT --save_path $SAVE_RESULT_PATH
--pre_trained $PRETRAINED_MODEL_PATH
```

### train
```
python train.py --train_root $TRAIN_DATA_ROOT --test_root $VALIDATION_DATA_ROOT
```