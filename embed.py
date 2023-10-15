import json
import torch
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple, Optional
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RAND_SEED = 42

def load_model_tokenizer(model_name: str, 
                         quantization_config: Optional[BitsAndBytesConfig] = None,
                         **kwargs) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, return_tensors="pt")
    return model, tokenizer

def load_toxic_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, names=['goal', 'target'])

def extract_embedding_dataset(df: pd.DataFrame, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[np.ndarray]:
    embed_list = []
    df['concat'] = df['goal'] #  + "\n" + df['target']
    texts = df['concat'].to_numpy()
    for g in tqdm(texts, total=texts.shape[0]):
        inputs = tokenizer(g, return_tensors='pt')
        outputs = model(**inputs)
        embed_list += [outputs.hidden_states[-1].detach().cpu().numpy()]
    return embed_list

def write_embeds(path: str, toxic_embeds: List[np.ndarray], clean_embeds: List[np.ndarray] = []):
    data = {
        "toxic_embeds": [],
        "clean_embeds": []
    }
    for emb in toxic_embeds:
        data['toxic_embeds'] += [emb.tolist()]

    if clean_embeds is not None:
        for emb in clean_embeds:
            data['clean_embeds'] += [emb.tolist()]

    with open(path, "w") as f:
        json.dump(data, f)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-m", "--model-name", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-c", "--clean-data", type=str, default="")
    parser.add_argument("-i", "--instruction-column", type=str, default="instruction")
    parser.add_argument("-s", "--split", type=str, default="train")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model, tokenizer = load_model_tokenizer(args.model_name, quantization_config=nf4_config, output_hidden_states=True, output_attentions=True)
    df = load_toxic_dataset(args.path)
    toxic_embeds = extract_embedding_dataset(df, model, tokenizer)

    clean_embeds = []
    if args.clean_data != "":
        dataset = load_dataset(args.clean_data, split=args.split)

        # randomly select data from clean dataset
        # dataset = dataset.filter(lambda d: "?" not in d['instruction'])
        dataset = dataset.shuffle(seed=RAND_SEED)
        dataset = dataset.select(range(len(toxic_embeds)))

        pd_dataset = pd.DataFrame(dataset)
        pd_dataset['goal'] = pd_dataset[args.instruction_column]
        clean_embeds = extract_embedding_dataset(pd_dataset, model, tokenizer)
    write_embeds(args.output, toxic_embeds, clean_embeds=clean_embeds) 
    # test_str = "What is the tallest mountain in the world?"
    # inputs = tokenizer(test_str, return_tensors="pt")
    # outputs = model(**inputs)
    # print(outputs.hidden_states[0].shape)