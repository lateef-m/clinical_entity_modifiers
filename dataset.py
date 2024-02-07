import config
import torch

class Semeval_Dataset:
    def __init__(self, data, tokenizer, max_len):
        self.contexts = data['context'].values
        self.mentions = data['disorder_mention'].values

#         for i in config.LABEL_COLUMNS:
#             globals()[f'self.{i}'] = data[f'{i}'].values
        self.negations = data['negation'].values
        self.courses = data['course'].values
        self.severity = data['severity'].values
        self.uncertainty = data['uncertainty'].values
        self.conditionals = data['conditional'].values
        self.subjects = data['subject'].values
        self.genetics = data['genetic'].values
#         self.body_location = data['body_location'].values

        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, item):
        context = str(self.contexts[item])
        mention = str(self.mentions[item])

        encoding = self.tokenizer.encode_plus(
            context,
            mention,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',          #  pad_to_max_length=True,  >>>  deprecated
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
            'context': context,
            'mention': mention,
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': {
            'target_negation': torch.tensor(self.negations[item], dtype=torch.long),
            'target_course': torch.tensor(self.courses[item], dtype=torch.long),
            'target_severity': torch.tensor(self.severity[item], dtype=torch.long),
            'target_uncertainty': torch.tensor(self.uncertainty[item], dtype=torch.long),
            'target_conditional': torch.tensor(self.conditionals[item], dtype=torch.long),
            'target_subject': torch.tensor(self.subjects[item], dtype=torch.long),
            'target_genetic': torch.tensor(self.genetics[item], dtype=torch.long),
#             'body_location': torch.tensor(self.body_location[item], dtype=torch.long),
            }
        }
    