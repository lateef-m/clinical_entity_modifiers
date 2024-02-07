# clinical_entity_modifiers

## Requirments
```
```

## Preprocess:
Preprocess the raw ShARe data set (a dir with train, devel, and test subdir).
```
python preprocess_share.py \
--input_dir /path/to/the/data/ \
--output_dir /path/to/the/data/processed/ \
-sep \|
```

## Train:
Train and save best model.
```
python train.py \
!python train_share.py \
--input_dir /path/to/the/data/processed/ \
--log_dir /path/to/logs/ \
--modifiers negation severity subject uncertainty course conditional genetic \
--epochs 10 \
-lr 2e-5 \
-b 64 -vb 32 -max 144 \
--random_seed 42 \
--save_state_dict False \
--cache_dir modifires_model_share.pt \
-m bio_bert_uncased
```

## Test:
Evaluate on the test dataset. Run test.py to get unweited accuracy and macro-F1. Run test_micro to get micro-F1
```
python test.py \
--input_dir /path/to/the/data/processed/ \
--log_dir /path/to/logs/ \
--modifiers negation severity subject uncertainty course conditional genetic \
-vb 32 -max 144 \
--cache_dir modifires_model_share.pt
```

```
python test_micro.py \
--input_dir /path/to/the/data/processed/ \
--log_dir /path/to/logs/ \
--modifiers negation severity subject uncertainty course conditional genetic \
-vb 32 -max 144 \
--cache_dir modifires_model_share.pt
```

## Predict:
Generate a prediction for each test file to be evaluated by the ShARe task evaluation script
```
python predict_each_file.py \
--input_dir /data/user/lateef11/bioNLP/github/data/processed/ \
--log_dir /data/user/lateef11/bioNLP/github/logs/ \
--output_dir /data/user/lateef11/bioNLP/github/data/ \
--modifiers negation severity subject uncertainty course conditional genetic \
-vb 32 -max 144 \
--cache_dir modifires_model_share.pt
```
