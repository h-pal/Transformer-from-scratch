from trans_model import Transformers
import torch
from torch import nn
from datasets import load_dataset
import os
from torch.utils.data import Dataset as PDataset
import string
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from collections import Counter
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import torch.nn.functional as F
import wandb

stopwords.words('english')
nltk.download('words') 

wandb.init(project="Practise", name="Transformers-MLM")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

punct = string.punctuation + '0123456789'
english_words = set(words.words())
stops = set(stopwords.words("english"))

def pre_process(data_obj):
    raw_text = data_obj["text"]
    raw_text = "".join(raw_text).lower()

    clean_text = raw_text.translate(str.maketrans('', '', punct))
    clean_text = " ".join(clean_text.split())
    clean_text = clean_text.split()
    clean_text = [word for word in clean_text if word in english_words and word not in stops]
    return clean_text

def make_vocab(data_list, vocab_size = 10000):
    counter_obj = Counter(data_list)
    vocab = {word: idx for idx, (word, count) in enumerate(counter_obj.most_common(vocab_size))}
    vocab["<START>"] = len(vocab)
    vocab["<END>"] = len(vocab)
    vocab["<MSK>"] = len(vocab)
    vocab["<UNK>"] = len(vocab)
    return vocab

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="/uufs/chpc.utah.edu/common/home/tasdizen-group1/tutorial/data/")


train_ds = pre_process(ds["train"])
val_ds = pre_process(ds["validation"])
test_ds = pre_process(ds["test"])

vocab = make_vocab(train_ds)


class DataWrapper(PDataset):
    def __init__(self, data_ds, vocab, context_window = 16):
        self.raw = data_ds
        self.vocab = vocab
        self.context = context_window - 2
    
    def __len__(self):
        return len(self.raw) - self.context

    def __getitem__(self, idx):
        x = self.raw[idx:idx + self.context]
        y = self.raw[idx + self.context]

        y_vec = [vocab["<START>"]] + [self.vocab.get(word, vocab["<UNK>"]) for word in x] + [vocab["<END>"]]
        x_vec = [vocab["<MSK>"] if i in random.sample(range(1, len(y_vec) - 1), min(2, len(y_vec) - 2)) else val for i, val in enumerate(y_vec)]

        return {"input": torch.tensor(x_vec, dtype=torch.long), "label": torch.tensor(y_vec, dtype=torch.long)}

seq_len = 16
batch_size = 1024

train_dataset = DataWrapper(train_ds, vocab, seq_len)
test_dataset = DataWrapper(test_ds, vocab, seq_len)
val_dataset = DataWrapper(val_ds, vocab, seq_len)

train_loader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, drop_last = True)


input_dim = 256
ffn_dim = 2048
num_encoder_blocks = 6
num_decoder_block = 6
vocab_size = len(vocab)
epochs = 5
val_every = 10000
top_k = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformers(num_encoder_blocks, num_decoder_block, vocab_size, input_dim, ffn_dim, h=16, d = 256).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)


for epoch in range(epochs):
    epoch_loss, epoch_perp, epoch_top_k, steps = 0, 0, 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        model.train()
        steps += 1
        x = batch['input'].to(device)
        y = batch['label'].to(device)
        preds = model(x,y)
        preds_copy = preds.view(-1, vocab_size)
        y_copy = y.view(-1)
        # y_copy = torch.flatten(y)
        # preds_copy = preds.view(batch_size * seq_len, vocab_size)
        loss = loss_fn(preds_copy, y_copy)
        perp = torch.exp(loss)
        if top_k is not None:
            probs = F.softmax(preds, dim=-1) 
            _, top_k_preds = torch.topk(probs, k=top_k, dim=-1)  
            top_k_accuracy = (top_k_preds == y.unsqueeze(-1)).any(dim=-1).float().mean()
            epoch_top_k += top_k_accuracy.item()

        epoch_loss += loss.item()
        epoch_perp += perp.item()
        wandb.log({"train_loss": loss.item() , "step": steps})
        wandb.log({"train_perpexility": perp.item() , "step": steps})
        wandb.log({"Train-Top-5-Accu": top_k_accuracy.item(), "step": steps})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    

        if steps % val_every == 0:
            print("validating")
            val_loss_all, val_perp_all, val_topk, val_steps = 0, 0, 0, 0
            model.eval()
            for batch in val_loader:
                val_steps+=1
                x_val = batch['input'].to(device)
                y_val = batch['label'].to(device)
                with torch.no_grad():
                    preds_val = model(x_val,y_val)
                val_preds_copy = preds_val.view(-1, vocab_size)
                val_y_copy = y_val.view(-1)
                val_loss = loss_fn(val_preds_copy, val_y_copy)
                val_perp = torch.exp(val_loss)
                
                if top_k is not None:
                    probs = F.softmax(preds_val, dim=-1) 
                    _, top_k_preds = torch.topk(probs, k=top_k, dim=-1) 
                    top_k_accuracy = (top_k_preds == y_val.unsqueeze(-1)).any(dim=-1).float().mean()
                    val_topk += top_k_accuracy.item()

                val_loss_all += val_loss.item()
                val_perp_all += val_perp.item()

            wandb.log({"val_loss": val_loss_all / val_steps, "step": steps})
            wandb.log({"val_perpexility": val_perp_all / val_steps, "step": steps})
            wandb.log({"Val-Top-5-Accu": val_topk / val_steps, "step": steps})
            print("="*50)
            print(f"Val Loss: {val_loss_all / steps } Val Perpexility: {val_perp_all / steps} Val Top 5 Accu: {val_topk / steps}")

    