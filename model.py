import config
import numpy as np
from transformers import AutoModel    
import torch
import torch.nn as nn

    
class Semeval_Model(nn.Module):
    def __init__(self, labelEncoder_meta_data_dict, dropout: float = 0.2):  # num_uncer_classes
        super(Semeval_Model, self).__init__()
#         for i in config.LABEL_COLUMNS:
#             globals()[f"self.num_{i}_classes"] = len(labelEncoder_meta_data_dict[i].classes_.tolist())

        self.num_negation_classes = len(labelEncoder_meta_data_dict["negation"].classes_.tolist())
        self.num_course_classes = len(labelEncoder_meta_data_dict["course"].classes_.tolist())
        self.num_severity_classes = len(labelEncoder_meta_data_dict["severity"].classes_.tolist())
        self.num_uncertainty_classes = len(labelEncoder_meta_data_dict["uncertainty"].classes_.tolist())
        self.num_conditional_classes = len(labelEncoder_meta_data_dict["conditional"].classes_.tolist())
        self.num_subject_classes = len(labelEncoder_meta_data_dict["subject"].classes_.tolist())
        self.num_genetic_classes = len(labelEncoder_meta_data_dict["genetic"].classes_.tolist())
#         self.num_body_location_classes = len(labelEncoder_meta_data_dict["body_location"].classes_.tolist())
        

        self.num_of_modifiers = len(labelEncoder_meta_data_dict)

        self.model = AutoModel.from_pretrained(config.MODEL)


#         for i in config.LABEL_COLUMNS:
#             globals()[f"self.{i}_classifier"] = nn.Sequential(
#                 nn.Dropout(p=dropout),
#                 nn.Linear(self.model.config.hidden_size, globals()[f"self.num_{i}_classes"]),
#             )

        self.negation_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_negation_classes),
        )
        
        self.course_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_course_classes),
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_severity_classes),
        )
        
        self.uncertainty_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_uncertainty_classes),
        )
        
        self.conditional_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_conditional_classes),
        )
        
        self.subject_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_subject_classes),
        )
        self.genetic_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.config.hidden_size, self.num_genetic_classes),
        )
        
#         self.body_location_classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(self.model.config.hidden_size, self.num_body_location_classes),
#         )

    def forward(self, input_ids, token_type_ids, attention_mask):
#         _, pooled_features = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
#                                         attention_mask=attention_mask)
        model_outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)
        pooled_features = model_outputs[1]
        
        negation_output = self.negation_classifier(pooled_features)
        course_output = self.course_classifier(pooled_features)
        severity_output = self.severity_classifier(pooled_features)
        uncertainty_output = self.uncertainty_classifier(pooled_features)
        conditional_output = self.conditional_classifier(pooled_features)
        subject_output = self.subject_classifier(pooled_features)
        genetic_output = self.genetic_classifier(pooled_features)
#         body_location_output = self.body_location_classifier(pooled_features)
        
        return {
            'negation_output': negation_output,
            'course_output': course_output,
            'severity_output': severity_output,
            'uncertainty_output': uncertainty_output,
            'conditional_output': conditional_output,
            'subject_output': subject_output,
            'genetic_output': genetic_output,
#             'body_location_output': body_location_output
                }

#         output={}
#         for i in config.LABEL_COLUMNS:
#             output[f'{i}_output'] = globals()[f"self.{i}_classifier"](pooled_features)
#         return output
        # return {f'{i}_output': globals()[f"self.{i}_classifier"](pooled_features) for i in config.LABEL_COLUMNS}
