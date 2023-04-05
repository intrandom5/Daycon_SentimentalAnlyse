import torch
import torch.nn as nn
from transformers import AutoModel


class DialogueEmotionClassifier(nn.Module):
    def __init__(self, model_name, max_speakers=100, num_labels=7):
        super(DialogueEmotionClassifier, self).__init__()
        self.model_name = model_name
        self.max_speakers = max_speakers
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(model_name)
        self.SpeakerEmbeddingLayer = nn.Embedding(max_speakers, self.bert.config.hidden_size)
        self.Linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, data: dict, mask_position):
        input_ids_embedding = self.bert.embeddings(data["input_ids"])
        speaker_ids_embedding = self.SpeakerEmbeddingLayer(data["speaker_ids"])
        embedding = input_ids_embedding + speaker_ids_embedding

        attention_mask = self.bert.get_extended_attention_mask(data["attention_mask"], data["input_ids"].shape)
        bert_output = self.bert.encoder(embedding, attention_mask=attention_mask).last_hidden_state
        output = self.Linear(bert_output)

        final_output = []
        for i, b in enumerate(output):
            mask_output = torch.index_select(b, dim=0, index=torch.LongTensor(mask_position[i]))
            final_output.append(mask_output)
        final_output = torch.cat(final_output, dim=0)

        return final_output
