#!/usr/bin/env bash

python MetricCnnTrainingInference.py --brain_id=100610 --input_dir=../Brains --output_dir=../Checkpoints --gpu_device=1 --epoch_num=1001 --learning_rate=1e-2 --terminating_loss=1e0 --checkpoint_save_frequency=100 