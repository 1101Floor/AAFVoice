# This file has the configurations of the experiments.
import os
import torch
import multiprocessing

TRAIN_DATA_DIR = os.path.join(os.path.expanduser("~"), "D:/Data/LibriSpeech_train/train-clean-360")
TEST_DATA_DIR = os.path.join(os.path.expanduser("~"), "D:/Data/LibriSpeech_test/test-clean")

TRAIN_DATA_CSV = ""
TEST_DATA_CSV = ""

SAVED_BINPOINT_PATH = "save/points.pt"
SAVED_MODEL_PATH = "save/AFSC+ResCNN.pt"

'''
设置LSTM参数
'''
# Hidden size of LSTM layers.
LSTM_HIDDEN_SIZE = 512
# Number of LSTM layers.
LSTM_NUM_LAYERS = 3
# Whether to use bi-directional LSTM.
BI_LSTM = True
# If false, use last frame of LSTM inference as aggregated output;
# if true, use mean frame of LSTM inference as aggregated output.
FRAME_AGGREGATION_MEAN = True
# Sequence length of the sliding window.
SEQ_LEN = 100  # 2.5 seconds
'''
设置GRU参数
'''
# Hidden size of GRU layers.
GRU_HIDDEN_SIZE = 512
# Number of GRU layers.
GRU_NUM_LAYERS = 3
# Whether to use bi-directional LSTM.
BI_GRU = True
'''
设置Transfomer参数
'''
# If true, we use transformer instead of LSTM.
USE_TRANSFORMER = True
# Dimension of transformer layers.
TRANSFORMER_DIM = 64
dimension = 512
# Number of encoder layers for transformer
TRANSFORMER_ENCODER_LAYERS = 3
# Number of heads in transformer layers.
TRANSFORMER_HEADS = 8
'''
其他设置
'''
N_MFCC = 80
# Alpha for the triplet loss.
TRIPLET_ALPHA = 0.1
# How many triplets do we train in a single batch.
BATCH_SIZE = 4
# Learning rate.
LEARNING_RATE = 0.0001
# Save a model to disk every these many steps.
SAVE_MODEL_FREQUENCY = 50000
# Number of steps to train.
TRAINING_STEPS = 20000
# Whether we are going to train with SpecAugment.
SPECAUG_TRAINING = False
# Parameters for SpecAugment training.
SPECAUG_FREQ_MASK_PROB = 0.3
SPECAUG_TIME_MASK_PROB = 0.3
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5
# Whether to use full sequence inference or sliding window inference.
USE_FULL_SEQUENCE_INFERENCE = True
# Sliding window step for sliding window inference.
SLIDING_WINDOW_STEP = 50  # 1.6 seconds
# Number of triplets to evaluate for computing Equal Error Rate (EER).
# Both the number of positive trials and number of negative trials will be
# equal to this number.
NUM_EVAL_TRIPLETS = 5000
# Step of threshold sweeping for computing Equal Error Rate (EER).
EVAL_THRESHOLD_STEP = 0.001
# Number of processes for multi-processing.
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)
# Wehther to use GPU or CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
