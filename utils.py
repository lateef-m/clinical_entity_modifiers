import torch
from tqdm import tqdm
import config
import numpy as np
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # print(input)
        # print(target)

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        # print(input)
        # print(target)
        # print('--------------- 1 --------------------')
        logpt = F.log_softmax(input, dim=1)
        # print(logpt)
        logpt = logpt.gather(1,target)
        # print(logpt)
        # print(':::::::::::::')
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        # print(logpt)
        # print(pt)
        # print('--------------- 2 --------------------')

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            # at = self.alpha.gather(0,target)
            # print('********')
            print(at)
            # print('********')
            logpt = logpt * Variable(at.data)
            # print('--------------- end --------------------')

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def get_loss(model_output, ground_truth):        
    loss = 0.0
    for i in config.LABEL_COLUMNS:
        # loss_fn = FocalLoss(gamma=5)            
        # loss_fn = nn.CrossEntropyLoss(weight = weight_dict[i].to(config.device))
        loss_fn = nn.CrossEntropyLoss()
        locals()[f'{i}_loss'] = loss_fn(model_output[f'{i}_output'], ground_truth[f'target_{i}'].to(config.device))            
#             locals()[f'{i}_loss'] = nn.CrossEntropyLoss(model_output[f'{i}_output'], ground_truth[f'target_{i}'].to(config.device))
        loss += locals()[f'{i}_loss']

    loss = loss / len(config.LABEL_COLUMNS)

    losses_dict = {}
    for i in config.LABEL_COLUMNS:
        losses_dict[i] = locals()[f'{i}_loss']

    return loss, losses_dict

def train_fn(data_loader, model, optimizer, device, scheduler, n_examples):
    model.train()

    final_loss = 0

    # creat a local variable for every target and init. it with 0 (# neg_correct_pred = 0)
    # correct_pred_vars = []
    for i in config.LABEL_COLUMNS:
        locals()[f'{i}_correct_pred'] = 0
        # correct_pred_vars.append()

    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        model_output = model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask
                             )
#         print(locals())
        # negation_output = model_output['negation_output']

        # loss, losses_train = model.get_loss(model_output, batch['labels'])
        loss, losses_train = get_loss(model_output, batch['labels'])

        for i in config.LABEL_COLUMNS:
            # torch.max() return (values, indices)
            locals()[f'{i}_pred'] = torch.max(model_output[f'{i}_output'], dim=1)[1]
            # locals()[f'{i}_pred'] = torch.max(locals()[f'{i}_output'], dim=1)[1]
        # _, neg_preds = torch.max(negation_output, dim=1)

        for i in config.LABEL_COLUMNS:
            locals()[f'{i}_correct_pred'] += torch.sum(locals()[f'{i}_pred'] == batch['labels'][f'target_{i}'].to(device))
        # neg_correct_pred += torch.sum(neg_preds == batch['target_negation'])

        #         mcc = metrics.matthews_corrcoef(y_true, y_pred)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

    # accuracies = np.zeros(len(config.LABEL_COLUMNS))
    # for j, i in zip(np.arange(len(config.LABEL_COLUMNS)), config.LABEL_COLUMNS):
    #     accuracies[j] = locals()[f'{i}_correct_pred'] / float(n_examples)
    #
    # return accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4], accuracies[5], accuracies[6], final_loss/len(data_loader)
    accuracies_dict = {}
    for i in config.LABEL_COLUMNS:
        accuracies_dict[i] = locals()[f'{i}_correct_pred'] / float(n_examples)

    return accuracies_dict, final_loss/len(data_loader)


def eval_fn(data_loader, model, device, n_examples):
    model.eval()
    final_loss = 0

    # creat a local variable for every target and init. it with 0 (# neg_correct_pred = 0)
    for i in config.LABEL_COLUMNS:
        locals()[f'{i}_correct_pred'] = 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        model_output = model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask
                             )
#         print(model_output)
        
        # loss, losses_eval = model.get_loss(model_output, batch['labels'])
        loss, losses_eval = get_loss(model_output, batch['labels'])

        for i in config.LABEL_COLUMNS:
            # torch.max() return (values, indices)
            locals()[f'{i}_pred'] = torch.max(model_output[f'{i}_output'], dim=1)[1]
            # locals()[f'{i}_pred'] = torch.max(locals()[f'{i}_output'], dim=1)[1]
        # _, neg_preds = torch.max(negation_output, dim=1)

        for i in config.LABEL_COLUMNS:
            locals()[f'{i}_correct_pred'] += torch.sum(locals()[f'{i}_pred'] == batch['labels'][f'target_{i}'].to(device))

        final_loss += loss.item()

    # accuracies = np.zeros(len(config.LABEL_COLUMNS))
    # for j, i in zip(np.arange(len(config.LABEL_COLUMNS)), config.LABEL_COLUMNS):
    #     accuracies[j] = locals()[f'{i}_correct_pred'] / float(n_examples)
    #
    # return accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4], accuracies[5], accuracies[6], final_loss / len(data_loader)
    accuracies_dict = {}
    for i in config.LABEL_COLUMNS:
        accuracies_dict[i] = locals()[f'{i}_correct_pred'] / float(n_examples)

    return accuracies_dict, final_loss/len(data_loader)
