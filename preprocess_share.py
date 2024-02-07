import config
# import config_withArgs as config

import joblib
import os
import re
import pandas as pd
from sklearn import preprocessing, model_selection
import argparse
import csv


from pathlib import Path
# models_dir = f'{config.}/models/'
Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def SemEval_pipe_to_custom_pipe(input_dir, output_dir, _sep):
    """ Generate traning dev and test data from .pipe and .text files.
    
    This function create a customized conll file for each pipe and text file of the SemEval dataset.
    Each line in the customized file will contains "context|mention tokens|modifier1|modifier2|...|modifierN"
    
    Args:
        input_dir: folder that contains .pipe and .text files.
        output_dir: folder that will contains generated data
    
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for n in ['ecg', 'radiology', 'echo']:
    print(os.listdir(input_dir))
    print()
    for n in os.listdir(input_dir):

        dir1 = os.path.join(input_dir, n) # <directory path/discharge>
   
        # Go through each subdirectory.
        # os.walk return  tuple of 3 items: ('current diretory path','subdirs','files in current diretory')
        for subdir in os.walk(dir1): 
        # for subdir in listdir(dir1):
            # print(subdir)   
            for file in subdir[2]:
                # print(file)

                if '.pipe' in file:
                    pipe_file_path = os.path.join(subdir[0], file)
                    text_file = os.path.splitext(file)[0] + '.text'
                    # print(text_file)
                    if text_file in subdir[2]:   # check if a text file is exist for the same pipe file (some text files are missed)
                        with open (pipe_file_path) as f:
                            pipe_data = f.readlines() # pipe_data: list['line1', 'line2',...,'lineN']

                        with open (os.path.splitext(pipe_file_path)[0] + '.text') as f:
                            text = f.read()

                        f = open(output_dir + os.path.splitext(file)[0] + '_custom.pipe', 'w', encoding='utf-8')
                        # print(out_dir +'/'+file+'_all.conll')
                        for i in pipe_data:
                            span = i.split('|')[1]
                            start = int(span.split('-')[0])
                            end = int(span.split('-')[-1])                            
                            mention = text[start:end]
                            context_len_char_before = 200
                            context_len_char_after = 50
                            # " ".join(text.split()) --> clean text with no newline or extra spaces
                            context = " ".join(text[max(0,start - context_len_char_before):
                                                    end + context_len_char_after].split()) # max to consider begenning of the file
                            
                            context = context.replace("|", ".")
                            offsets = [i.split('-') for i in span.split(',')]
                            starts = [int(i[0]) for i in offsets]
                            ends = [int(i[1]) for i in offsets]
                            mentions = [text[starts[i]:ends[i]] for i in range(len(offsets))]
   
                            doc_name = i.split('|')[0]
                            negation = i.split('|')[3]                            
                            subject = i.split('|')[5]
                            uncertainty = i.split('|')[7]
                            course = i.split('|')[9]
                            severity = i.split('|')[11]                            
                            conditional = i.split('|')[13]
                            genetic = i.split('|')[15]
                            # body_location = i.split('|')[17]

                            mentions_str = ' '.join(str(item) for item in mentions)
                            f.write(f'{doc_name}{_sep}{context}{_sep}{mentions_str}{_sep}{negation}{_sep}{severity}{_sep}{subject}{_sep}{uncertainty}{_sep}{course}{_sep}{conditional}{_sep}{genetic}\n')
                            
                        f.close()

                                    
def process_data_from_custom_pipe_folder(input_dir, output_dir, _type, _sep):
    # encoding='utf-8'

    # Preprocess train dataframe
    # columns = ['context', 'disorder_mention', 'negation','severity','subject','uncertainty','course','conditional', 'genetic', 'body_location']
    columns = ['doc_name','context','disorder_mention','negation','severity', 'subject', 'uncertainty','course','conditional','genetic']
    
    fileslist = os.listdir(input_dir) 
    fileslist_fullpath = [input_dir + file for file in fileslist if '_custom.pipe' in file]
    
    # print(fileslist)
    # print(fileslist_fullpath)

    #read each file into pandas
    # df_list = [pd.read_table(file, sep=_sep, names=columns, encoding='utf-8') for file in fileslist_fullpath]
    df_list = [pd.read_csv(file, sep=_sep, names=columns, quoting=csv.QUOTE_NONE, encoding='utf-8') for file in fileslist_fullpath]
    
    #concatenate all the pandas together
    df_all_data = pd.concat(df_list)
    
    df_all_data.loc[df_all_data.conditional == True, 'conditional'] = 'true'
    df_all_data.loc[df_all_data.conditional == False, 'conditional'] = 'false'
    df_all_data.loc[df_all_data.genetic == True, 'genetic'] = 'true'
    df_all_data.loc[df_all_data.genetic == False, 'genetic'] = 'false'
    
    df_all_data = df_all_data[df_all_data.negation.isin(['yes', 'no'])]
    df_all_data = df_all_data[df_all_data.severity.isin(['unmarked', 'severe', 'moderate', 'slight'])]
    df_all_data = df_all_data[df_all_data.uncertainty.isin(['yes','no'])]
    df_all_data = df_all_data[df_all_data.course.isin(['unmarked', 'increased', 'decreased', 'improved','resolved','worsened'])]
    df_all_data = df_all_data[df_all_data.conditional.isin(['false', 'true'])]
    df_all_data = df_all_data[df_all_data.genetic.isin(['false', 'true'])]
    df_all_data = df_all_data[df_all_data.subject.isin(['patient', 'family_member', 'other'])]

    df_all_data.to_csv(output_dir + f'{_type}_df.csv', sep=_sep, encoding='utf-8', index=False)
    
    if _type == "train":
        # Encode all the target classes using a loop
        labelEncoder_dict = {}      # dictionary that have labelEncoder for every target column
        # for i in config.LABEL_COLUMNS:
        for i in columns[3:]:
            le = preprocessing.LabelEncoder()
            df_all_data.loc[:, i] = le.fit_transform(df_all_data[i])
            labelEncoder_dict[i] = le

            print(f'{i}: {len(le.classes_.tolist())}')
        print(labelEncoder_dict)
        joblib.dump(labelEncoder_dict, output_dir + "share_labelEncoder_meta.bin")
        
    else:
        labelEncoder_meta_data = joblib.load(output_dir + "share_labelEncoder_meta.bin")
        for i in columns[3:]:
            df_all_data.loc[:, i] = labelEncoder_meta_data[i].transform(df_all_data[i])
        
    df_all_data.to_csv(output_dir + f'{_type}_encoded_df.csv', sep=_sep, encoding='utf-8', index=False)

    return df_all_data


if __name__ == '__main__':
    
    input_dir = config.INPUT_DIR
    output_dir = config.OUTPUT_DIR
    _sep = config.SEP
    
    SemEval_pipe_to_custom_pipe(input_dir + "train2/", output_dir + "train/", _sep)
    SemEval_pipe_to_custom_pipe(input_dir + "devel/", output_dir + "devel/", _sep)
    SemEval_pipe_to_custom_pipe(input_dir + "test/", output_dir + "test/", _sep)
    
    process_data_from_custom_pipe_folder(output_dir + "train/", output_dir, _type = "train", _sep=_sep)
    process_data_from_custom_pipe_folder(output_dir + "devel/", output_dir, _type = "devel", _sep=_sep)
    process_data_from_custom_pipe_folder(output_dir + "test/", output_dir, _type = "test", _sep=_sep)

    print('Done pre-processing ...')
