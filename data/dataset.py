from torch.utils.data import Dataset
import torch


class EmoDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        super(EmoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sentences = self.tokenizer(
            list(sentences),
            padding=True,
            return_tensors="pt"
        )

        self.labels = labels
        self.label2num = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }
        self.num2label = {v: k for k, v in self.label2num.items()}

    def __len__(self):
        return len(self.sentences['input_ids'])

    def __getitem__(self, idx):
        output = {
            "input_ids": self.sentences["input_ids"][idx],
            "attention_mask": self.sentences["attention_mask"][idx],
            "labels": self.label2num[self.labels[idx]]
        }
        return output


class EmoDataset_v1(Dataset):
    def __init__(self, csv, tokenizer, max_length=256):
        super(EmoDataset_v1, self).__init__()
        self.csv = pd.read_csv(csv)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        self.data = self.construct_dialogue(max_length)

    def construct_dialogue(self, max_length):
        total_input_ids = []
        total_speaker_ids = []
        total_attention_masks = []
        total_mask_positions = []
        total_labels = []
        dialogue_ids = list(self.csv["Dialogue_ID"].unique())

        for dialogue_id in dialogue_ids:
            dialogue = self.csv[self.csv["Dialogue_ID"] == dialogue_id]
            speaker_dict = {speaker: i for i, speaker in enumerate(dialogue["Speaker"].unique())}
            tokenized_comments = []
            speaker_ids = []
            mask_positions = []
            labels = []
            for speaker, comment, label in zip(dialogue["Speaker"], dialogue["Utterance"], dialogue["Target"]):
                tokenized_comment = self.tokenizer.encode(comment, add_special_tokens=False) + [
                    self.tokenizer.mask_token_id, self.tokenizer.sep_token_id
                ]
                label = self.label2id[label]

                if len(tokenized_comments) + len(tokenized_comment) > max_length:
                    # 길이가 넘어가면 그대로 추가.
                    pad_length = max_length - len(tokenized_comments)
                    attention_mask = [1 for _ in range(len(tokenized_comments))] + [0 for _ in range(pad_length)]
                    speaker_ids += [0 for _ in range(pad_length)]
                    tokenized_comments += [self.tokenizer.pad_token_id for _ in range(pad_length)]

                    total_attention_masks.append(attention_mask)
                    total_input_ids.append(tokenized_comments)
                    total_speaker_ids.append(speaker_ids)
                    total_mask_positions.append(mask_positions)
                    total_labels.append(labels)
                    # 그 다음 대화록부터 다시 데이터 구성.
                    tokenized_comments = tokenized_comment
                    speaker_ids = [speaker_dict[speaker] for _ in range(len(tokenized_comment))]
                    mask_positions = [len(tokenized_comment) - 2]
                    labels = [label]
                else:
                    mask_positions.append(len(tokenized_comments) + len(tokenized_comment) - 2)
                    tokenized_comments += tokenized_comment
                    speaker_ids += [speaker_dict[speaker] for _ in range(len(tokenized_comment))]
                    labels.append(label)

            pad_length = max_length - len(tokenized_comments)
            attention_mask = [1 for _ in range(len(tokenized_comments))] + [0 for _ in range(pad_length)]
            speaker_ids += [0 for _ in range(pad_length)]
            tokenized_comments += [self.tokenizer.pad_token_id for _ in range(pad_length)]

            total_attention_masks.append(attention_mask)
            total_input_ids.append(tokenized_comments)
            total_speaker_ids.append(speaker_ids)
            total_mask_positions.append(mask_positions)
            total_labels.append(labels)
        return {
            "input_ids": total_input_ids,
            "attention_mask": total_attention_masks,
            "speaker_ids": total_speaker_ids,
            "mask_position": total_mask_positions,
            "labels": total_labels
        }

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "speaker_ids": self.data["speaker_ids"][idx],
            "mask_position": self.data["mask_position"][idx],
            "labels": self.data["labels"][idx]
        }

def data_collator(batch):
    input_ids = []
    attention_mask = []
    speaker_ids = []
    mask_position = []
    labels = []
    for b in batch:
        input_ids.append(torch.LongTensor(b["input_ids"]))
        attention_mask.append(torch.LongTensor(b["attention_mask"]))
        speaker_ids.append(torch.LongTensor(b["speaker_ids"]))
        mask_position.append(torch.LongTensor(b["mask_position"]))
        labels.append(torch.LongTensor(b["labels"]))
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "speaker_ids": torch.stack(speaker_ids, dim=0),
        "mask_position": mask_position,
        "labels": labels
    }
