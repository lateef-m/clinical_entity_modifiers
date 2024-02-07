import numpy as np
import torch
import joblib
import pandas as pd

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import utils
from model import Semeval_Model

from pathlib import Path
import logging
from datetime import date

######## logging #########
today = str(date.today())
log_dir = f'{config.LOG_DIR}'
# models_dir = f'{config.}/models/'
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path('models/').mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s--%(filename)s--%(message)s')
file_handler = logging.FileHandler(f'{log_dir}{config.TRAINED_MODEL}_{today}_train.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


if __name__ == '__main__':

    train_df = pd.read_csv(f'{config.INPUT_DIR}train_encoded_df.csv', sep='|', encoding='utf-8')
    eval_df = pd.read_csv(f'{config.INPUT_DIR}devel_encoded_df.csv', sep='|', encoding='utf-8')
    
    # train_df = train_df.sample(500)
    # eval_df = eval_df.sample(500)

    train_dataset = dataset.Semeval_Dataset(train_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
    eval_dataset = dataset.Semeval_Dataset(eval_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=0)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=0)
    logger.debug(len(train_data_loader))

    # A dictionary that have labelEncoder for every target column
    labelEncoder_meta_data = joblib.load(f"{config.INPUT_DIR}share_labelEncoder_meta.bin")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(device)
    model = Semeval_Model(labelEncoder_meta_data)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, correct_bias=False, weight_decay=0.01)
    total_steps = len(train_data_loader) * config.EPOCHS
    logger.debug(f'total epochs: {config.EPOCHS}')
    logger.debug(f'total_steps: {total_steps}')

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_accuracy = 0
    best_loss = np.inf

    for epoch in range(config.EPOCHS):
        logger.debug(f"EPOCH {epoch + 1}/{config.EPOCHS}")
        train_accuracies_dict, train_loss = utils.train_fn(train_data_loader, model, optimizer, device, scheduler,
                                                              len(train_df))
        eval_accuracies_dict, eval_loss = utils.eval_fn(eval_data_loader, model, device, len(eval_df))

        logger.debug(f"Train Loss = {train_loss} Valid Loss = {eval_loss}")
        for i in config.LABEL_COLUMNS:
            logger.debug(f"Train accuracy <{i}>= {train_accuracies_dict[i]} -- Valid accuracy <{i}>= {eval_accuracies_dict[i]}")

#         if eval_loss < best_loss:
#             logger.info("saving model ...")
#             torch.save(model, f'models/{config.TRAINED_MODEL}')
#             best_loss = eval_loss
        
        if sum(eval_accuracies_dict.values())/len(eval_accuracies_dict) > best_accuracy:
            logger.info("saving model ...")
            
            # save full model
            # torch.save(model, f'models/{config.TRAINED_MODEL}_e{epoch}.pt')
            torch.save(model, f'models/{config.TRAINED_MODEL}')
            # save model state_dict
            if config.SAVE_STATE_DICT:
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                # torch.save(checkpoint, f"models/{config.TRAINED_MODEL}_e{epoch}.pt.tar")
                torch.save(checkpoint, f"models/{config.TRAINED_MODEL}.tar")
            
            best_accuracy = sum(eval_accuracies_dict.values())/len(eval_accuracies_dict)
            