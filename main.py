from tqdm import tqdm
import pandas as pd
import yaml

import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import EmoDataset, EmoDataset_v1, data_collator
from models import DialogueEmotionClassifier


def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    epochs = config["epoch"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]

    train_dataset = EmoDataset_v1("train_data.csv", tokenizer, max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

    valid_dataset = EmoDataset_v1("valid_data.csv", tokenizer, max_length=256)
    valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=data_collator)

    model = DialogueEmotionClassifier(config["model_name"])

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        model.train()
        for data in tqdm(train_loader):
            label = data.pop("labels")
            mask_position = data.pop("mask_position")
            data = {k: v.to(device) for k, v in data.items()}
            label = torch.cat(label)

            output = model(data, mask_position)
            loss = criterion(output, label)
            train_losses.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        cor = 0
        val_loss = 0
        for data in tqdm(valid_loader):
            label = data.pop("labels")
            mask_position = data.pop("mask_position")
            data = {k: v.to(device) for k, v in data.items()}
            label = torch.cat(label)

            with torch.no_grad():
                output = model(data)
            loss = criterion(output, label)
            val_loss += loss.cpu()
            output = torch.argmax(output, dim=-1)
            for lab, out in zip(label, output):
                if lab == out:
                    cor += 1
        valid_losses.append(val_loss / len(valid_loader))
        print("valid accuracy :", cor / len(valid_dataset))


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
