from transformers import AutoTokenizer
import argparse
import torch

parser = argparse.ArgumentParser(description='Capture training parameters')
parser.add_argument('-e', '--epochs', type=int, default=2, help='number of training cycles') #, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-vb', '--val_batch_size', type=int, default=16, help='batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('-m', '--model_type', type=str, default='bert', help='transformer model to use')

parser.add_argument('-d', '--input_dir', type=str, default='None', help='dir of the final preprocesded (train/dev/test) datasets')
parser.add_argument('-l', '--log_dir', type=str, help='path to log dir') #, required=True)
parser.add_argument('-o', '--output_dir', type=str, help='path to output folders')
parser.add_argument('-df', '--data_folder', type=str, default='None', help='dir of the pipe files')

# optional arguments - these are the ones most likely to need changing
parser.add_argument('-c', '--cache_dir', type=str, default='oud_modifires_model.pt', help='location to save/load model files')
parser.add_argument('-r', '--random_seed', type=int, default=42, help='provide a value for deterministic code')

# parser.add_argument('-d', '--dataset', type=str, default='None')
parser.add_argument('-mod', '--modifiers', type=str, nargs='*',
                    default=['negation', 'subject' , 'uncertainty'],
                    help='list of label columns (modifiers)')
parser.add_argument('-t', '--task', type=str, default='None')
parser.add_argument('-sep', '--col_sep', type=str, default='|', help='column seperator')
parser.add_argument('-st', '--save_state_dict', type=bool, help='Save state_dict. It will be large. Only needed for trasfer learning to another dataset')

# optional arguments that likely won't need modification
parser.add_argument('-max', '--max_sequence_length', type=int, default=128, help='max tokens per sample, up to 512')

args = parser.parse_args()

MAX_LEN = args.max_sequence_length
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.val_batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
RANDOM_SEED = args.random_seed
TRAINED_MODEL = args.cache_dir
# DATASET = args.dataset
# TRAINING_FOLDER = ''
# VALID_FOLDER = ''
INPUT_DIR = args.input_dir
LOG_DIR = args.log_dir
OUTPUT_DIR = args.output_dir
LABEL_COLUMNS = args.modifiers
TASK = args.task
SEP = args.col_sep
SAVE_STATE_DICT = args.save_state_dict

if args.model_type == 'bio_bert':
    MODEL = 'dmis-lab/biobert-base-cased-v1.1'
if args.model_type == 'bio_bert_uncased':
    MODEL = 'dmis-lab/biobert-base-cased-v1.1'
if args.model_type == 'bert':
    MODEL = 'bert-base-cased'
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL,
    do_lower_case=True,
    use_fast=True)

# LABEL_COLUMNS = ['negation', 'doctime', 'illicitDrugUse', 'subject' , 'severity', 'uncertainty']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")