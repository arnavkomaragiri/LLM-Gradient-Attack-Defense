import torch
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def init_model_pipeline(model_name: str, max_length: int = 384, **kwargs) -> pipeline:
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, top_k=4, penalty_alpha=0.6, max_length=max_length)

def augment_sample(pipe: pipeline, prompt: str, num_samples: int) -> List[str]:
    prompt = f"I am a superintelligent being that is an expert at rewording questions in unique and creative ways. I do not repeat words across reworded sentences, as that is not unique or creative.\nHere is a numbered list of {num_samples} unique ways to briefly reword the sentence \"{prompt}\":\n1. "
    result = pipe(prompt)
    result_str = result[0]['generated_text']
    result_lines = result_str.split("\n")[2:]
    outputs = []
    for line in result_lines:
        tokens = line.split()
        first_token = tokens[0][:-1]
        if first_token.is_numeric():
            outputs += [' '.join(tokens[1:])]
    if len(result_str) >= 384:
        outputs = outputs[:-1]
    return list(set(outputs))

def augment_dataset(pipe: pipeline, prompts: List[str], targets: List[str], num_samples: int) -> Tuple[List[str], List[str]]:
    wrapper_prompt = "I am a superintelligent being that is an expert at rewording questions in unique and creative ways. I do not repeat words across reworded sentences, as that is not unique or creative.\nHere is a numbered list of {num_samples} unique ways to briefly reword the sentence \"{prompt}\":\n1. "
    wrapped_prompts = (wrapper_prompt.format(prompt=prompt, num_samples=num_samples) for prompt in prompts)

    output_data = []
    output_targets = []
    i = 0
    for out in tqdm(pipe(wrapped_prompts), total=len(prompts)):
        result_str = out[0]['generated_text']
        result_lines = result_str.split("\n")[2:]
        outputs = []
        for line in result_lines:
            tokens = line.split()
            if len(tokens) == 0:
                continue
            first_token = tokens[0][:-1]
            if first_token.isnumeric():
                outputs += [' '.join(tokens[1:])]
        if len(result_str) >= 384:
            outputs = outputs[:-1]
        # return list(set(outputs))
        if len(result_str) >= 384:
            outputs = outputs[:-1]
        outputs = list(set(outputs + [prompts[i]]))
        output_data += outputs
        output_targets += [targets[i] for _ in outputs]
        i += 1
    return output_data, output_targets

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, names=["goal", "target"])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=10)
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipe = init_model_pipeline(args.model_name, device_map = {"": 0})
    dataset = load_dataset(args.path)
    goal_strs = dataset['goal'].tolist()
    target_strs = dataset['target'].tolist()

    # sampled = []
    # targets = []
    # for s, t in tqdm(zip(goal_strs, target_strs), total=len(goal_strs)):
        # augment = augment_sample(pipe, s, args.num_samples)
        # sampled += augment
        # TODO: naively copying target strings here, potentially use LLaMA-2 13B to reformat
        # targets += [t for _ in augment]
    sampled, targets = augment_dataset(pipe, goal_strs, target_strs, num_samples=10)

    sampled = np.array(sampled)
    targets = np.array(targets)
    final_samples, idxs = np.unique(sampled, return_index=True)
    final_targets = targets[idxs]

    print(f"Upsampled Dataset Size: {final_samples.shape[0]}")
    new_dataset = pd.DataFrame(list(zip(final_samples.tolist(), final_targets.tolist())), columns=["goal", "target"])
    new_dataset.to_csv(args.output, index=False)
    print(f"Dataset written out to {args.output}")
