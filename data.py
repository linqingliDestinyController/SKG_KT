# data.py
# SKG-KT Dataset, Collator, and prompt formatting utilities

from typing import List, Union, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from prompts import SKGKT_SYSTEM_PROMPT


# -----------------------------
# Prompt Formatting
# -----------------------------
def correct_to_str(correct: Union[bool, None]):
    return "na" if correct is None else "true" if correct else "false"


def standards_to_str(standards: List[str], sep: str):
    return "None" if not standards else sep.join([f"{idx + 1}) {kc}" for idx, kc in enumerate(standards)])


def apply_annotations(sample: dict, texts, apply_na: bool = True):
    records = []
    student_id = sample["student_id"]
    exercise_logs = sample["exercises_logs"]
    is_corrects = sample["is_corrects"]
    KCs = sample["KCs"]
    Descriptions = sample["Description"]

    for i in range(len(exercise_logs)):
        if len(KCs[i]) == 0:
            continue
        record_i = {
            "student_id": student_id,
            "exercise_id": exercise_logs[i],
            "exercise_text": texts[int(exercise_logs[i])]["exercise_desc"],
            "correct": True if is_corrects[i] else False,
            "kcs": KCs[i],
            "kc_descriptions": Descriptions[i],
        }
        records.append(record_i)
    return records


def get_record_text(records: List[dict], kg, turn_idx: int, include_labels: bool = False, tag_wrapper: bool = True):
    lines = []
    max_history = 4
    start_idx = max(0, turn_idx - max_history)

    for idx in range(start_idx, turn_idx):
        turn = records[idx]
        lines.append(f"History: Student id is {turn['student_id']}")
        lines.append(f"History: Exercise id is {turn['exercise_id']}")
        lines.append(f"History: Exercise Text is {turn['exercise_text']}")
        lines.append(f"History: Correct: {correct_to_str(turn['correct'])}")
        lines.append(f"History: Knowledge Components: {standards_to_str(turn['kc_descriptions'], ' ')}")

    current_turn = records[turn_idx]
    lines.append(f"Current Student id is {current_turn['student_id']}")
    lines.append(f"Current Exercise id is {current_turn['exercise_id']}")
    lines.append(f"Current Exercise Text is {current_turn['exercise_text']}")

    e_kg = kg[current_turn["exercise_id"]]
    prompt = "\n".join(lines)

    lines_kg = []
    for triple in e_kg:
        if triple.get("relation") != "Individual":
            head = triple.get("head", "")
            relation = triple.get("relation", "")
            tail = triple.get("tail", "")
            lines_kg.append(f"[{head}] --{relation}--> [{tail}]")

    prompt += "\n[BEGIN STUDENT KNOWLEDGE GRAPH]\n" + "\n".join(lines_kg) + "\n[END STUDENT KNOWLEDGE GRAPH]"
    if tag_wrapper:
        prompt = "[BEGIN STUDENT RECORDS]\n" + prompt + "\n[END STUDENT RECORDS]"
    return prompt


def skgkt_system_prompt(args):
    return SKGKT_SYSTEM_PROMPT


def skgkt_user_prompt(records: List[dict], kg, turn_idx: int, kc: Optional[str], args):
    prompt = ""
    prompt += get_record_text(records, kg, turn_idx=turn_idx, include_labels=args.prompt_inc_labels)
    prompt += f"\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


# -----------------------------
# Dataset
# -----------------------------
class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SKGKTDatasetPacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, texts, kgs, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0

        for idx, sample in data.iterrows():
            records = apply_annotations(sample, texts)
            for turn_id, turn in enumerate(records):
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": skgkt_system_prompt(args)},
                        {"role": "user", "content": skgkt_user_prompt(records, kgs, turn_id, None, args)},
                    ],
                    tokenize=False,
                )

                kc_conts = [
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": kc},
                            {"role": "assistant", "content": "\n"},
                        ],
                        tokenize=False,
                    )
                    for kc in turn["kc_descriptions"]
                ]
                # strip template prefix
                kc_conts = [" " + cont.split("user<|end_header_id|>\n\n")[1] for cont in kc_conts]
                prompt = prompt + "".join(kc_conts)

                self.data.append(
                    {
                        "student_idx": idx,
                        "prompt": prompt,
                        "label": int(turn["correct"]),
                        "kcs": turn["kc_descriptions"],
                    }
                )

        print(f"{failed} / {len(data)} records failed processing")
        print(f"Number of data points: {len(self.data)}")


class SKGKTCollatorPacked:
    """DDP-safe collator: keeps all tensors on CPU; loss moves them to model.device."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        if isinstance(batch, dict):
            batch = [batch]

        prompts = [sample["prompt"] for sample in batch]
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tok.input_ids
        batch_size, max_seq_len = input_ids.shape

        eos_idxs = []
        for seq_idx in range(batch_size):
            idxs = (input_ids[seq_idx] == self.tokenizer.eos_token_id).nonzero(as_tuple=False).squeeze(-1).cpu()
            if idxs.ndim == 0:
                idxs = idxs.unsqueeze(0)
            eos_idxs.append(idxs)

        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].bool()
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)

        for seq_idx in range(batch_size):
            if eos_idxs[seq_idx].numel() < 2:
                context_end_idx = max_seq_len
            else:
                context_end_idx = int(eos_idxs[seq_idx][1].item())

            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0

            start_idx = context_end_idx + 1
            for end_idx_t in eos_idxs[seq_idx][3::2]:
                end_idx = int(end_idx_t.item())
                if start_idx >= max_seq_len or end_idx >= max_seq_len:
                    break

                new_len = end_idx - start_idx + 1
                if new_len > 0:
                    position_ids[seq_idx, start_idx:end_idx + 1] = torch.arange(
                        context_end_idx,
                        context_end_idx + new_len
                    )

                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1

                start_idx = end_idx + 1

        last_idxs = pad_sequence([idxs[3::2] - 1 for idxs in eos_idxs], batch_first=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.unsqueeze(1),
            "position_ids": position_ids,
            "last_idxs": last_idxs,
            "num_kcs": torch.LongTensor([len(sample["kcs"]) for sample in batch]),
            "labels": torch.Tensor([sample["label"] for sample in batch]),
            "meta_data": batch,
        }
