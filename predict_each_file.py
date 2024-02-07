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
                # print('----------2')
                # pipe_file_path = '/data/user/lateef11/bioNLP/semeval/test_data/test1/00174-002042-Copy1.pipe'
                # file = '00174-002042-Copy1.pipe'
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
#                     f.write(
#                         f'{doc_name}|{span}|{DD_CUI}|{line[3]}|{line[4]}|{line[5]}|{line[6]}|{line[7]}|'
#                         f'{line[8]}|{line[9]}|{line[10]}|{line[11]}|{line[12]}|{conditional}|{line[14]}|'
#                         f'{line[15]}|{line[16]}|{line[17]}|{line[18]}')
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
    
#     genetic_errors_df = pd.DataFrame(genetic_errors)
#     print(genetic_errors_df.head())
          


if __name__ == "__main__":

    # test_df = joblib.load('/data/user/lateef11/bioNLP/semeval/semeval_data/fromGuergana/test_df.bin')
    # test_dataset = dataset.SemevalDataset(test_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)

    # input_dir = '/data/user/lateef11/bioNLP/semeval/test_data/test1/'
    # output_dir = '/data/user/lateef11/bioNLP/semeval/test_data/test_output/'
#     input_dir = '/data/user/lateef11/bioNLP/semeval/semeval_data/fromGuergana/goldstandard_entities/test/discharge/'
#     output_dir = '/data/user/lateef11/bioNLP/semeval/semeval_optimized/all_modifiers/test_output_pipe_offset_context/'
#     output_dir = '/data/user/lateef11/bioNLP/semeval/semeval_optimized/all_modifiers/test_output_no_s2/'
    # predict_each_file(input_folder, output_folder)
#     predict_each_file(input_dir, output_dir, True)
    predict_each_file(config.INPUT_DIR, config.OUTPUT_DIR, True)




# import joblib
# import torch
# from tqdm import tqdm
# from sklearn import metrics
#
# # import config
# import config_withArgs as config
# import dataset
# from model import SemevalModel
# import glob
# import pandas as pd
# import os
#
#
# def predict_each_file(input_folder, output_folder):
#     labelEncoder_meta_data_dict = joblib.load("meta.bin")
#     # All files ending with .pipe
#     all_test_files = (glob.glob(input_folder + "*.pipe"))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model = torch.load(config.TRAINED_MODEL)
#     model.to(device)
#     model.eval()
#
#     final_loss = 0
#
#     # creat a local variable for every target and init. it with 0 (# neg_correct_pred = 0)
#     for i in config.LABEL_COLUMNS:
#         locals()[f'{i}_correct_pred'] = 0
#         locals()[f'{i}_predictions'] = []
#         locals()[f'{i}_real_values'] = []
#
#     all_columns = 'DD_DocName|DD_Spans|DD_CUI|negation|Cue_NI|subject|Cue_Experiencer|uncertainty|Cue_UI|course|Cue_CC|severity|Cue_SV|conditional|Cue_CO|genetic|Cue_GC|Norm_BL|Cue_BL'
#     columns = all_columns.split('|')
#     cols = ['negation', 'severity', 'course', 'uncertainty', 'conditional', 'subject', 'genetic']
#
#     # for n in ['ecg', 'radiology', 'echo']:
#     #     #     for n in os.listdir(train_directory):
#     #
#     #
#     #     dir1 = os.path.join(directory, n)  # <directory path/discharge>
#     #     # print("1")
#     #     # print(dir1)
#     #
#     #     # Go through each subdirectory.
#     #     # os.walk return  tuple of 3 items: ('current diretory path','subdirs','files in current diretory')
#     #     for subdir in os.walk(dir1):
#             # for subdir in listdir(dir1):
#             #             print(subdir)
#     all_files = os.listdir(input_folder)
#     for file in all_files:
#         print(file)
#         print('----------')
#
#         if '.pipe' in file:
#             pipe_file_path = os.path.join(input_folder, file)
#             text_file = os.path.splitext(file)[0] + '.text'
#             # print(text_file)
#             # check if a text file is exist for the same pipe file (some text files are missed)
#             if text_file in all_files:
#                 print('----------2')
#                 with open(pipe_file_path) as f:
#                     pipe_data = f.readlines()  # pipe_data: list['line1', 'line2',...,'lineN']
#
#                 with open(os.path.splitext(pipe_file_path)[0] + '.text') as f:
#                     text = f.read()
#
#                 f = open(output_folder + file, 'w', encoding='utf-8')
#
#                 for i in pipe_data:
#                     line = i.split('|')
#                     span = i.split('|')[1]
#                     start = int(span.split('-')[0])
#                     end = int(span.split('-')[-1])
#                     mention = text[start:end]
#                     context_len_char_before = 200
#                     context_len_char_after = 50
#                     # " ".join(text.split()) --> clean text with no newline or extra spaces
#                     context = " ".join(text[max(0,
#                                                 start - context_len_char_before):end + context_len_char_after].split())  # max to consider begenning of the file
#                     doc_name = i.split('|')[0]
#                     DD_CUI = i.split('|')[2]
#                     negation = i.split('|')[3]
#                     subject = i.split('|')[5]
#                     uncertinty = i.split('|')[7]
#                     course = i.split('|')[9]
#                     severity = i.split('|')[11]
#                     conditional = i.split('|')[13]
#                     genetic = i.split('|')[15]
#                     body_location = i.split('|')[17]
#
#                     #                             f.write(f'{context}|{mention}|{negation}|{severity}\n')
#                     # f.write(f'{doc_name}|{context}|{mention}|{negation}|{severity}|{subject}|{uncertinty}|{course}|{conditional}|{genetic}|{body_location}\n')
#
# # all_columns = 'DD_DocName|DD_Spans|DD_CUI|negation|Cue_NI|subject|Cue_Experiencer|uncertainty|Cue_UI|course|Cue_CC|severity|Cue_SV|conditional|Cue_CO|genetic|Cue_GC|Norm_BL|Cue_BL'
#                     f.write(
#                         f'{doc_name}|{span}|{DD_CUI}|{negation}|{line[4]}|{subject}|{line[6]}|{uncertinty}|'
#                         f'{line[8]}|{course}|{line[10]}|{severity}|{line[12]}|{conditional}|{line[14]}|'
#                         f'{genetic}|{line[16]}|{body_location}|{line[18]}')
#                     # f.write(
#                     #     f'{doc_name}\t{context}\t{mention}\t{negation}\t{severity}\t{subject}\t{uncertinty}\t{course}\t{conditional}\t{genetic}\t{body_location}\n')
#                 f.close()
#
#     for file in all_test_files:
#         test_df = pd.read_table(file, sep='|', names=columns)
#         test_dataset = dataset.SemevalDataset(test_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
#         data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE)
#         print(test_df.head(2))
#         print('-' * 100)
#
#         with torch.no_grad():
#             for batch in tqdm(data_loader, total=len(data_loader)):
#                 input_ids = batch['input_ids'].to(device)
#                 token_type_ids = batch['token_type_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#
#
#         # with torch.no_grad():
#         #     data = test_dataset[0]
#         #     for k, v in data.items():
#         #         data[k] = v.to(device).unsqueeze(0)
#         #     tag, pos, _ = model(**data)
#         #
#         #     print(
#         #         enc_tag.inverse_transform(
#         #             tag.argmax(2).cpu().numpy().reshape(-1)
#         #         )[:len(tokenized_sentence)]
#         #     )
#
#
#
#     with torch.no_grad():
#         for batch in tqdm(data_loader, total=len(data_loader)):
#             input_ids = batch['input_ids'].to(device)
#             token_type_ids = batch['token_type_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#
#             model_output = model(input_ids=input_ids,
#                                  token_type_ids=token_type_ids,
#                                  attention_mask=attention_mask
#                                  )
#
#             loss, losses_train = model.get_loss(model_output, batch['labels'])
#
#             for i in config.LABEL_COLUMNS:
#                 # torch.max() return (values, indices)
#                 locals()[f'{i}_pred'] = torch.max(model_output[f'{i}_output'], dim=1)[1]
#                 locals()[f'{i}_predictions'].extend(locals()[f'{i}_pred'])
#                 locals()[f'{i}_real_values'].extend(batch['labels'][f'target_{i}'])
#
#             for i in config.LABEL_COLUMNS:
#                 locals()[f'{i}_correct_pred'] += torch.sum(locals()[f'{i}_pred'] == batch['labels'][f'target_{i}'].to(device))
#
#             final_loss += loss.item()
#
#     for i in config.LABEL_COLUMNS:
#         if i == 'conditional' or i == 'genetic':
#             continue
#         locals()[f'{i}_predictions'] = torch.stack(locals()[f'{i}_predictions']).cpu()
#         locals()[f'{i}_real_values'] = torch.stack(locals()[f'{i}_real_values']).cpu()
#
#         print(f"{i} Report:")
#         print(metrics.classification_report(locals()[f'{i}_predictions'], locals()[f'{i}_real_values'],
#                                             target_names=labelEncoder_meta_data_dict[f"{i}"].classes_,
#                                             digits=4, zero_division=1))
#         print('-' * 100)
#
#
#     # return neg_predictions, real_neg_values,real_illicitDrugUse_values, real_subj_values, encode_neg.classes_, encode_sev.classes_
#     # return float(neg_correct_pred) / n_examples, float(sev_correct_pred) / n_examples, final_loss / len(data_loader)
#
#
# if __name__ == "__main__":
#     # train_df = joblib.load('train_df.bin')
#     # eval_df = joblib.load('/data/user/lateef11/bioNLP/semeval/semeval_data/fromGuergana/eval_df.bin')
# #     eval_df = joblib.load('test_df.bin')
#     #     eval_df = eval_df[:100]
#     test_df = joblib.load('/data/user/lateef11/bioNLP/semeval/semeval_data/fromGuergana/test_df.bin')
#
#     # eval_dataset = dataset.SemevalDataset(eval_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
#     # test_dataset = dataset.SemevalDataset(test_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
#     # eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4)
#     # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
#
#     # predict(eval_data_loader, len(eval_df))
#     # print('-' * 60)
#     input_folder = '/data/user/lateef11/bioNLP/semeval/test_data/test1/'
#     output_folder = '/data/user/lateef11/bioNLP/semeval/test_data/test_output/'
#     predict_each_file(input_folder, output_folder)


# import joblib
# import torch
# from tqdm import tqdm
# from sklearn import metrics

# # import config
# import config_withArgs as config
# import dataset
# from model import SemevalModel
# import glob
# import pandas as pd
# import os
# import json
# import numpy as np

# # import spacy
# import medspacy

# def get_sentence_from_offsets_with_cumulative_sum(offsets, sentences, tuple_list, context_side):
#     final_sentences =[]
#     context_sentences_offsets = set()
    
#     #loops through entities and gets their sentence
#     count=0
#     offset_start = offsets.min()
#     offset_end = offsets.max()
#     for i in range(len(tuple_list)):
#         if offset_start>=tuple_list[i][0] and offset_start<=tuple_list[i][1] and offset_end>=tuple_list[i][0] and offset_end<=tuple_list[i][1]:
#             context_sentences_offsets.add(i)
#             count+=1

#         #this will check if the entity spans over multiple sentences.
#         #if this happens then something bad happened like the tokenizer tokenized wrong
#         if count>1:
#             print('count!=1: \n ** Found more than one context')
    
#     # from old Sentence_Tokenizer
#     for i in context_sentences_offsets:
#         # no_context
#         if context_side == 'no':
#             final_sentences.append((sentences[i]).strip())
        
#         # both_sides_context
#         if context_side == 'both':
#             if i==0 and len(sentences)==1:
#                 final_sentences.append((sentences[i]).strip())
#             elif i==0 and len(sentences)>1:
#                 final_sentences.append((sentences[i]).strip()+' '+(sentences[i+1]).strip())
#             elif i==len(tuple_list)-1:
#                 final_sentences.append((sentences[i-1]).strip()+' '+(sentences[i]).strip())
#             else:
#                 final_sentences.append((sentences[i-1]).strip()+' '+(sentences[i]).strip()+' '+(sentences[i+1]).strip())

#         # 2_previous_context
#         if context_side == '2_previous':
#             if i==0 and len(sentences)==1:
#                 final_sentences.append((sentences[i]).strip())
#             elif i==0 and len(sentences)==2:
#                 final_sentences.append((sentences[i-1]).strip()+' '+(sentences[i]).strip())
#             else:
#                 final_sentences.append((sentences[i-2]).strip()+' '+(sentences[i-1]).strip()+' '+(sentences[i]).strip())

#     return ' '.join(final_sentences)
# #     return (final_sentences)

# def predict_each_file(input_folder, output_folder, print_errors = False):
    
#     #loads scispacy tokenizer
#     # nlp = spacy.load("en_core_web_sm")
#     #loads medspacy tokenizer
#     nlp = medspacy.load()

#     labelEncoder_meta_data_dict = joblib.load("meta.bin")

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = torch.load(config.TRAINED_MODEL)
#     model.to(device)
#     model.eval()

#     final_loss = 0
#     total_errors = 0
    
#     negation_errors = []
#     subject_errors = []
#     uncertainty_errors = []
#     course_errors = []
#     severity_errors = []
#     conditional_errors = []
#     genetic_errors = []

#     # all_columns = 'DD_DocName|DD_Spans|DD_CUI|negation|Cue_NI|subject|Cue_Experiencer|uncertainty|Cue_UI|course|Cue_CC|severity|Cue_SV|conditional|Cue_CO|genetic|Cue_GC|Norm_BL|Cue_BL'
#     # columns = all_columns.split('|')
#     # cols = ['negation', 'severity', 'course', 'uncertainty', 'conditional', 'subject', 'genetic']

#     # all_files = os.listdir(input_folder)
#     # for file in all_files:
#     #     print(file)
#     #     print('----------')
#     #
#     #     if '.pipe' in file:
#     #         pipe_file_path = os.path.join(input_folder, file)
#     #         text_file = os.path.splitext(file)[0] + '.text'
#     #         # print(text_file)
#     #         # check if a text file is exist for the same pipe file (some text files are missed)
#     #         if text_file in all_files:
#     #             print('----------2')
#     all_files = os.listdir(input_folder)
#     for file in all_files:
#         # print(file)
#         # print('----------')

#         if '.pipe' in file:
#             pipe_file_path = os.path.join(input_folder, file)
#             text_file = os.path.splitext(file)[0] + '.text'
#             # print(text_file)
#             # check if a text file is exist for the same pipe file (some text files are missed)
#             if text_file in all_files:
#                 # print('----------2')
#                 # pipe_file_path = '/data/user/lateef11/bioNLP/semeval/test_data/test1/00174-002042-Copy1.pipe'
#                 # file = '00174-002042-Copy1.pipe'
#                 with open(pipe_file_path) as f:
#                     pipe_data = f.readlines()  # pipe_data: list['line1', 'line2',...,'lineN']

#                 with open(os.path.splitext(pipe_file_path)[0] + '.text') as f:
#                     text = f.read()
                    
#                 doc = nlp(text)
#                 sentences = [i.text for i in doc.sents] #['This is a test sentence for show.','This is a second test sentence.','This is a third test sentence.']
# #                         sentences = [i.text.replace('\n',' ') for i in doc.sents] #['This is a test sentence for show.','This is a second test sentence.','This is a third test sentence.']
#                 #creates a cumulative sum of the offsets of each sentence
#                 cumulative_sum = np.cumsum([len(x) if x[-1]!='.' else len(x+" ") for x in sentences]) #array([34, 66, 97])
# #                         cumulative_sum = np.cumsum([len(x) for x in sentences]) #array([34, 66, 97])
# #                         cumulative_sum = np.cumsum([len(x+" ") for x in sentences]) #array([34, 66, 97])
#                 #creates a tuples off the offsets of the sentence
#                 tuple_list_sentences_offset = []
#                 for i in range(len(cumulative_sum)):
#                     if i-1 == -1:
#                         tuple_range = (0,cumulative_sum[i])
#                     else:
#                         tuple_range = (cumulative_sum[i-1], cumulative_sum[i])
#                     tuple_list_sentences_offset.append(tuple_range)  
#                     #tuple_list_sentences_offset = #[(0, 34), (34, 66),...]
#                 # print(tuple_list_sentences_offset)

#                 f = open(output_folder + file, 'w', encoding='utf-8')

#                 for i in pipe_data:

#                     line = i.split('|')
#                     span = i.split('|')[1]
#                     start = int(span.split('-')[0])
#                     end = int(span.split('-')[-1])
#                     mention = text[start:end]
# #                     context_len_char_before = 200
# #                     context_len_char_after = 50
# #                     # " ".join(text.split()) --> clean text with no newline or extra spaces
# #                     context = " ".join(text[max(0,
# #                                                 start - context_len_char_before):end + context_len_char_after].split())  # max to consider begenning of the file
#                     context = get_sentence_from_offsets_with_cumulative_sum(np.array([start,end]), sentences, tuple_list_sentences_offset, context_side='both')

#                     doc_name = i.split('|')[0]
#                     DD_CUI = i.split('|')[2]

#                     # genetic = labelEncoder_meta_data_dict['genetic'].transform([i.split('|')[15]])
#                     body_location = i.split('|')[17]

#                     encoding = config.TOKENIZER.encode_plus(
#                         context,
#                         mention,
#                         max_length=config.MAX_LEN,
#                         add_special_tokens=True,
#                         padding='max_length',  # pad_to_max_length=True,  >>>  deprecated
#                         truncation=True,
#                         return_attention_mask=True,
#                         return_token_type_ids=True,
#                         return_tensors='pt'
#                     )
#                     with torch.no_grad():
#                         # for batch in tqdm(pipe_data, total=len(pipe_data)):
#                         input_ids = encoding['input_ids'].to(device)
#                         token_type_ids = encoding['token_type_ids'].to(device)
#                         attention_mask = encoding['attention_mask'].to(device)

#                         model_output = model(input_ids=input_ids,
#                                              token_type_ids=token_type_ids,
#                                              attention_mask=attention_mask
#                                              )
#                         # print(labelEncoder_meta_data_dict['negation'].inverse_transform(
#                         #                 model_output['negation_output'].argmax(1).cpu().numpy().reshape(-1)
#                         #             ).item()#[:len(tokenized_sentence)]
#                         #     )

#                     negation = labelEncoder_meta_data_dict['negation'].inverse_transform(
#                         model_output['negation_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()
#                     subject = labelEncoder_meta_data_dict['subject'].inverse_transform(
#                         model_output['subject_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()
#                     uncertainty = labelEncoder_meta_data_dict['uncertainty'].inverse_transform(
#                         model_output['uncertainty_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()
#                     course = labelEncoder_meta_data_dict['course'].inverse_transform(
#                         model_output['course_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()
#                     severity = labelEncoder_meta_data_dict['severity'].inverse_transform(
#                         model_output['severity_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()
#                     conditional = labelEncoder_meta_data_dict['conditional'].inverse_transform(
#                         model_output['conditional_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()
#                     genetic = labelEncoder_meta_data_dict['genetic'].inverse_transform(
#                         model_output['genetic_output'].argmax(1).cpu().numpy().reshape(-1)
#                     ).item()

#                     #                     negation = i.split('|')[3]
#                     #                     subject = i.split('|')[5]
#                     #                     uncertinty = i.split('|')[7]
#                     #                     course = i.split('|')[9]
#                     #                     severity = i.split('|')[11]
#                     #                     conditional = i.split('|')[13]
#                     #                     genetic = i.split('|')[15]
#                     if print_errors:
# #                         pass
                        
#                         if negation != i.split('|')[3]:
#                             negation_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[3], 'pred_value': negation})
    
#                         if subject != i.split('|')[5]:
#                             subject_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[5], 'pred_value': subject})
    
#                         if uncertainty != i.split('|')[7]:
#                             uncertainty_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[7], 'pred_value': uncertainty})
    
#                         if course != i.split('|')[9]:
#                             course_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[9], 'pred_value': course})
    
#                         if severity != i.split('|')[11]:
#                             severity_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[11], 'pred_value': severity})
    
#                         if conditional != json.loads(i.split('|')[13].lower()):
#                             conditional_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[13], 'pred_value': conditional})
    
#                         if genetic != json.loads(i.split('|')[15].lower()):
#                             genetic_errors.append({'CONTEXT': context, 'MENTION': mention, 'gold_value': i.split("|")[15], 'pred_value': genetic})



# #                         if negation != i.split('|')[3] or subject != i.split('|')[5] or \
# #                                 uncertainty != i.split('|')[7] or course != i.split('|')[9] or \
# #                                 severity != i.split('|')[11] or conditional != json.loads(i.split('|')[13].lower()) or \
# #                                 genetic != json.loads(i.split('|')[15].lower()):

# #                             total_errors += 1
# #                             print(f'CONTEXT: {context}\n'
# #                                   f'MENTION: {mention}\n'
# #                                   f'negation: gold_value: {i.split("|")[3]}	pred_value: {negation}\n'
# #                                   f'subject: gold_value: {i.split("|")[5]}	pred_value: {subject}\n'
# #                                   f'uncertainty: gold_value: {i.split("|")[7]}	pred_value: {uncertainty}\n'
# #                                   f'course: gold_value: {i.split("|")[9]}	pred_value: {course}\n'
# #                                   f'severity: gold_value: {i.split("|")[11]}	pred_value: {severity}\n'
# #                                   f'conditional: gold_value: {i.split("|")[13]}	pred_value: {conditional}\n'
# #                                   f'genetic: gold_value: {i.split("|")[15]}	pred_value: {genetic}')
# #                             print('--------------------------------------------------')

#                     f.write(
#                         f'{doc_name}|{span}|{DD_CUI}|{negation}|{line[4]}|{subject}|{line[6]}|{uncertainty}|'
#                         f'{line[8]}|{course}|{line[10]}|{severity}|{line[12]}|{conditional}|{line[14]}|'
#                         f'{genetic}|{line[16]}|{body_location}|{line[18]}')
# #                     f.write(
# #                         f'{doc_name}|{span}|{DD_CUI}|{line[3]}|{line[4]}|{line[5]}|{line[6]}|{line[7]}|'
# #                         f'{line[8]}|{line[9]}|{line[10]}|{line[11]}|{line[12]}|{conditional}|{line[14]}|'
# #                         f'{line[15]}|{line[16]}|{line[17]}|{line[18]}')
#                 f.close()
# #     print(f'Total errors: {total_errors}')
#     print(f'Total errors: ')
# #     print(subject_errors)
#     print('negation_errors: ', len(negation_errors))
#     print('subject_errors: ', len(subject_errors))
#     print('uncertainty_errors: ', len(uncertainty_errors))
#     print('course_errors: ', len(course_errors))
#     print('severity_errors: ', len(severity_errors))
#     print('conditional_errors', len(conditional_errors))
#     print('genetic_errors', len(genetic_errors))
    
#     print('negation_errors:')
#     for i in negation_errors:
#         print(i)
#         print()
        
#     print('='*80)
#     print('subject_errors: ')
#     for i in subject_errors:
#         print(i)
#         print()
        
#     print('='*80)
#     print('uncertainty_errors: ')
#     for i in uncertainty_errors:
#         print(i)
#         print()
        
#     print('='*80)
#     print('course_errors: ')
#     for i in course_errors:
#         print(i)
#         print()
        
#     print('='*80)
#     print('severity_errors: ')
#     for i in severity_errors:
#         print(i)
#         print()

#     print('='*80)
#     print('conditional_errors')
#     for i in conditional_errors:
#         print(i)
#         print()
    
#     print('='*80)
#     print('genetic_errors')
#     for i in genetic_errors:
#         print(i)
#         print()
          


# if __name__ == "__main__":

#     # test_df = joblib.load('/data/user/lateef11/bioNLP/semeval/semeval_data/fromGuergana/test_df.bin')
#     # test_dataset = dataset.SemevalDataset(test_df, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN)
#     # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)

#     # input_folder = '/data/user/lateef11/bioNLP/semeval/test_data/test1/'
#     # output_folder = '/data/user/lateef11/bioNLP/semeval/test_data/test_output/'
#     input_folder = '/data/user/lateef11/bioNLP/semeval/semeval_data/fromGuergana/goldstandard_entities/test/discharge/'
#     output_folder = '/data/user/lateef11/bioNLP/semeval/semeval_optimized/all_modifiers/test_output_pipe_both_sides_context/'
#     # predict_each_file(input_folder, output_folder)
#     predict_each_file(input_folder, output_folder, True)
