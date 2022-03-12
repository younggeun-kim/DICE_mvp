import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


class Dataset():

    def __init__(self, classification_df, name='train'):
        super(Dataset).__init__()
        self.name = name
        self.tweet = []
        self.Y = []
        for index, rows in classification_df.iterrows():
            tweet = rows['keyword'] + rows['location'] + rows['text']
            self.tweet.append(''.join(tweet))
            if name == 'train' or self.name == 'valid':
                label = rows['target']
                self.Y.append(label)

        self.input_ids, self.attention_masks = tokenize(self.tweet)

    def __getitem__(self, index):

        tweet = self.tweet[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]

        if self.name == 'train' or self.name == 'valid':
            label = float(self.Y[index])
            return tweet, input_id, attention_masks, torch.as_tensor(label).long()
        else:
            return tweet, input_id, attention_masks

    def __len__(self):
        return len(self.tweet)

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