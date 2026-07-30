# Homework (Voice Anti-spoofing)


## About

The task is to implement anti-spoofing countermeasure against Logical Access (LA) attacks: by audiofile determine whether it is bonafide speech or spoof speech. 

## Dataset

The project uses the [ASVspoof 2019 dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset), Logical Access subset.

- `train` — training data;
- `dev` — validation data;
- `eval` — final evaluation data.

## Project structure

.
├── src/
│   ├── configs/ # hydra configurations
│   ├── datasets/ # dataset loading
│   ├── hw/ # EER calculation and grading
│   ├── logger/ # experiment logging
│   ├── loss/ # loss functions
│   ├── metrics/ # metrics
│   ├── model/ # LCNN model
│   ├── trainer/ # training and validation
│   ├── transforms/ # audio preprocessing
│   └── utils/


## Training

python train.py \
  datasets.train.audio_dir=/path/to/train/flac \
  datasets.train.protocol_path=/path/to/train_protocol.txt \
  datasets.val.audio_dir=/path/to/dev/flac \
  datasets.val.protocol_path=/path/to/dev_protocol.txt \
  dataloader.batch_size=16 \
  trainer.device=cuda \
  trainer.n_epochs=15

## Evaluation

The main metric is **Equal Error Rate (EER)**. Lower EER means better separation between bonafide and spoof speech.
