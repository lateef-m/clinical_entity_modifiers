import joblib
import torch
from tqdm import tqdm
from sklearn import metrics

import config
# import dataset
from model import Semeval_Model
# import glob
# import pandas as pd
import os
import json
import numpy as np

from pathlib import Path
import logging
from datetime import date

######## logging #########
today = str(date.today())
log_dir = f'{config.LOG_DIR}'
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(f'{config.OUTPUT_DIR}').mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s--%(filename)s--%(message)s')
file_handler = logging.FileHandler(f'{log_dir}{config.TRAINED_MODEL}_{today}_predictANDprintERRORS.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def predict_each_file(input_folder, output_folder, print_errors = False):
    
    # labelEncoder_meta_data_dict = joblib.load("data/labelEncoder_meta.bin")
    labelEncoder_meta_data_dict = joblib.load(f"{config.INPUT_DIR}share_labelEncoder_meta.bin")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model from state_dict
    # model = Semeval_Model(labelEncoder_meta_data_dict)
    # checkpoint = torch.load(f"models/{config.TRAINED_MODEL}.tar")
    # model.load_state_dict(checkpoint["state_dict"])
    
    # load full model
    model = torch.load(f"models/{config.TRAINED_MODEL}")
    model.to(device)
    model.eval()

    final_loss = 0
    total_errors = 0
    
    negation_errors = []
    subject_errors = []
    uncertainty_errors = []
    course_errors = []
    severity_errors = []
    conditional_errors = []
    genetic_errors = []

    # all_columns = 'DD_DocName|DD_Spans|DD_CUI|negation|Cue_NI|subject|Cue_Experiencer|uncertainty|Cue_UI|course|Cue_CC|severity|Cue_SV|conditional|Cue_CO|genetic|Cue_GC|Norm_BL|Cue_BL'
    # columns = all_columns.split('|')
    # cols = ['negation', 'severity', 'course', 'uncertainty', 'conditional', 'subject', 'genetic']


    all_files = os.listdir(input_folder + 'test/')
    for file in all_files:
        # print(file)
        # print('----------')

        if '.pipe' in file:
            pipe_file_path = os.path.join(input_folder, file)
            text_file = os.path.splitext(file)[0] + '.text'
            # print(text_file)
            # check if a text file is exist for the same pipe file (some text files are missed)
            if text_file in all_files:
                with open(pipe_file_path) as f:
                    pipe_data = f.readlines()  # pipe_data: list['line1', 'line2',...,'lineN']

                with open(os.path.splitext(pipe_file_path)[0] + '.text') as f:
                    text = f.read()
                    
                f = open(output_folder + file, 'w', encoding='utf-8')

                for i in pipe_data:

                    line = i.split('|')
                    span = i.split('|')[1]
                    start = int(span.split('-')[0])
                    end = int(span.split('-')[-1])
                    mention = text[start:end]
                    context_len_char_before = 200
                    context_len_char_after = 50
#                     # " ".join(text.split()) --> clean text with no newline or extra spaces
                    context = " ".join(text[max(0, start - context_len_char_before):end + context_len_char_after].split())  # max to consider begenning of the file

                    doc_name = i.split('|')[0]
                    DD_CUI = i.split('|')[2]
                    body_location = i.split('|')[17]

                    encoding = config.TOKENIZER.encode_plus(
                        context,
                        mention,
                        max_length=config.MAX_LEN,
                        add_special_tokens=True,
                        padding='max_length',  # pad_to_max_length=True,  >>>  deprecated
                        truncation=True,
                        return_attention_mask=True,
                        return_token_type_ids=True,
                        return_tensors='pt'
                    )
                    with torch.no_grad():
                        # for batch in tqdm(pipe_data, total=len(pipe_data)):
                        input_ids = encoding['input_ids'].to(device)
                        token_type_ids = encoding['token_type_ids'].to(device)
                        attention_mask = encoding['attention_mask'].to(device)

                        model_output = model(input_ids=input_ids,
                                             token_type_ids=token_type_ids,
                                             attention_mask=attention_mask
                                             )

                    negation = labelEncoder_meta_data_dict['negation'].inverse_transform(
                        model_output['negation_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()
                    subject = labelEncoder_meta_data_dict['subject'].inverse_transform(
                        model_output['subject_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()
                    uncertainty = labelEncoder_meta_data_dict['uncertainty'].inverse_transform(
                        model_output['uncertainty_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()
                    course = labelEncoder_meta_data_dict['course'].inverse_transform(
                        model_output['course_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()
                    severity = labelEncoder_meta_data_dict['severity'].inverse_transform(
                        model_output['severity_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()
                    conditional = labelEncoder_meta_data_dict['conditional'].inverse_transform(
                        model_output['conditional_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()
                    genetic = labelEncoder_meta_data_dict['genetic'].inverse_transform(
                        model_output['genetic_output'].argmax(1).cpu().numpy().reshape(-1)
                    ).item()


                    if print_errors:
#                         pass
                        
                        if negation != i.split('|')[3]:
                            negation_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[3], 'pred_value': negation})
    
                        if subject != i.split('|')[5]:
                            subject_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[5], 'pred_value': subject})
    
                        if uncertainty != i.split('|')[7]:
                            uncertainty_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[7], 'pred_value': uncertainty})
    
                        if course != i.split('|')[9]:
                            course_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[9], 'pred_value': course})
    
                        if severity != i.split('|')[11]:
                            severity_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[11], 'pred_value': severity})
    
                        if conditional != json.loads(i.split('|')[13].lower()):
                            conditional_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[13], 'pred_value': conditional})
    
                        if genetic != json.loads(i.split('|')[15].lower()):
                            genetic_errors.append({'doc#': doc_name, 'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[15], 'pred_value': genetic})

                    f.write(
                        f'{doc_name}|{span}|{DD_CUI}|{negation}|{line[4]}|{subject}|{line[6]}|{uncertainty}|'
                        f'{line[8]}|{course}|{line[10]}|{severity}|{line[12]}|{conditional}|{line[14]}|'
                        f'{genetic}|{line[16]}|{body_location}|{line[18]}')

                f.close()

#     logger.debug(f'Total errors: {total_errors}')
    logger.debug(f'Total errors: ')
    logger.debug(f'negation_errors: {len(negation_errors)}')
    logger.debug(f'subject_errors: {len(subject_errors)}')
    logger.debug(f'uncertainty_errors: {len(uncertainty_errors)}')
    logger.debug(f'course_errors: {len(course_errors)}')
    logger.debug(f'severity_errors: {len(severity_errors)}')
    logger.debug(f'conditional_errors: {len(conditional_errors)}')
    logger.debug(f'genetic_errors: {len(genetic_errors)}')
    
    logger.debug('negation_errors:')
    for i in negation_errors:
        logger.debug(i)
        logger.debug('\n')
        
    logger.debug('='*80)
    logger.debug('subject_errors: ')
    for i in subject_errors:
        logger.debug(i)
        logger.debug('\n')
        
    logger.debug('='*80)
    logger.debug('uncertainty_errors: ')
    for i in uncertainty_errors:
        logger.debug(i)
        logger.debug('\n')
        
    logger.debug('='*80)
    logger.debug('course_errors: ')
    for i in course_errors:
        logger.debug(i)
        logger.debug('\n')
        
    logger.debug('='*80)
    logger.debug('severity_errors: ')
    for i in severity_errors:
        logger.debug(i)
        logger.debug('\n')

    logger.debug('='*80)
    logger.debug('conditional_errors')
    for i in conditional_errors:
        logger.debug(i)
        logger.debug('\n')
    
    logger.debug('='*80)
    logger.debug('genetic_errors')
    for i in genetic_errors:
        logger.debug(i)
        logger.debug('\n')
              


if __name__ == "__main__":

    predict_each_file(config.INPUT_DIR, config.OUTPUT_DIR, True)
