import torch
import torch.nn.functional as F
import pandas as pd
import os
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def predict(model, test_loader, device,path):
    model.eval()
    target = []
    for batch_num, (captions, input_id, attention_masks) in enumerate(test_loader):


        input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
        output_dictionary = model(input_ids,
                                  token_type_ids=None,
                                  attention_mask=attention_masks,
                                  return_dict=True)

        predictions = F.softmax(output_dictionary['logits'], dim=1)

        _, top1_pred_labels = torch.max(predictions ,1)
        top1_pred_labels = top1_pred_labels.item()
        target.append(top1_pred_labels)


    make_csv(target,path)

def predict_input(model, input, device,path):

    model.eval()
    target = []
    input_id, attention_masks = tokenize([input])
    input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
    output_dictionary = model(input_ids,
                                  token_type_ids=None,
                                  attention_mask=attention_masks,
                                  return_dict=True)
    predictions = F.softmax(output_dictionary['logits'], dim=1)

    _, top1_pred_labels = torch.max(predictions ,1)
    top1_pred_labels = top1_pred_labels.item()
    target.append(top1_pred_labels)

    return target


def tokenize(sequences):
    input_ids = []
    attention_masks = []

    for seq in sequences:
        encoded_dict = tokenizer.encode_plus(
            seq,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

    



def make_csv(target,path):
    test = pd.read_csv(os.path.join(path,'test.csv'))
    my_submission = pd.DataFrame({'id': test.id, 'target': target})
    my_submission.to_csv(os.path.join(path,'submission.csv'), index=False)