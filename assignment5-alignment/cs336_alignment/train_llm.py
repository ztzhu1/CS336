from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Callable, List, Union
from unittest.mock import patch

import numpy as np
from openai import OpenAI
import pandas as pd
import torch
from tqdm.auto import tqdm, trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb

from cs336_alignment import drgrpo_grader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

with open(project_dir / "cs336_alignment" / "prompts" / "r1_zero.prompt", "r") as f:
    r1_zero_prompt = f.read()

with open(project_dir / "api_keys.json", "r") as f:
    api_keys = json.load(f)


def load_sft():
    path = project_dir.joinpath("data", "MATH", "sft.jsonl")
    sft_data = pd.read_json(
        path_or_buf=path,
        lines=True,
        dtype={"data_type": str, "data_index": int, "prompt": str, "response": str},
    )
    return sft_data


def make_prompt(
    input_text: Union[str, List[str]], prompt_type="r1_zero"
) -> Union[str, List[str]]:
    if not isinstance(input_text, str):
        return [make_prompt(t, prompt_type=prompt_type) for t in input_text]

    if prompt_type == "r1_zero":
        return r1_zero_prompt.format(question=input_text)
    else:
        raise NotImplementedError()


def make_sft_dataset(batch_size=1, indexes=None):
    path = project_dir.joinpath("data", "MATH", "train.jsonl")
    data = pd.read_json(path_or_buf=path, lines=True)

    with open(project_dir / "data" / "MATH" / "annot_CoT_prompt_en_v2.txt", "r") as f:
        pre_prompt = f.read()

    sft_data = load_sft()

    if len(sft_data) == 0:
        sft_data = pd.DataFrame(
            columns=["data_type", "data_index", "prompt", "response"]
        )
        start = 0
    else:
        start = sft_data.data_index.max() + 1

    # client = OpenAI(api_key=api_keys["deepseek"], base_url="https://api.deepseek.com")
    client = OpenAI(
        api_key=api_keys["chatanywhere"], base_url="https://api.chatanywhere.tech"
    )

    def get_response(index):
        prompt = data.iloc[index].problem
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a math Chain-of-Thought annotation expert. Generate the reasoning trace followed by an answer according to the requirements.",
                },
                {"role": "user", "content": pre_prompt + data.iloc[index].to_json()},
            ],
            stream=False,
        )
        response = response.choices[0].message.content
        return index, prompt, response

    pool = ThreadPoolExecutor()
    results = []
    if indexes is None:
        indexes = range(start, start + batch_size)
    for index in indexes:
        results.append(pool.submit(get_response, index))

    for result in results:
        index, prompt, response = result.result()
        sft_data.loc[index] = {
            "data_type": "train",
            "data_index": index,
            "prompt": prompt,
            "response": response,
        }
    path = project_dir.joinpath("data", "MATH", "sft.jsonl")
    # sft_data.to_json(path, lines=True, orient="records")
    save_jsonl(sft_data.to_dict(orient="records"), path, overwrite=True)


def check_sft_dataset():
    path = project_dir.joinpath("data", "MATH", "train.jsonl")
    data = pd.read_json(path_or_buf=path, lines=True)

    path = project_dir.joinpath("data", "MATH", "sft.jsonl")
    sft_data = pd.read_json(
        path_or_buf=path,
        lines=True,
        dtype={"data_type": str, "data_index": int, "prompt": str, "response": str},
    )
    rewards = {}
    for i in range(len(sft_data)):
        index = sft_data.iloc[i].data_index.item()
        reward = drgrpo_grader.r1_zero_reward_fn(
            sft_data.iloc[i].response, data.iloc[index].solution
        )
        if reward["reward"] != 1:
            rewards[index] = reward
    return rewards


def save_jsonl(lines: list[dict], path, overwrite=False):
    path = Path(path)
    if not overwrite:
        assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, line in enumerate(lines):
            json_line = json.dumps(line)
            if i == 0:
                f.write(json_line)
            else:
                f.write("\n" + json_line)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
    save_path=None,
    overwrite=False,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    """
    if save_path is not None:
        save_path = project_dir / "evaluation" / save_path
        if not overwrite:
            assert not save_path.exists()
    raw_prompts = prompts
    prompts = make_prompt(prompts)
    outputs = vllm_model.generate(prompts, sampling_params)
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        reward = reward_fn(response, ground_truths[i])
        result = {
            "problem": raw_prompts[i],
            "solution": ground_truths[i],
            "response": response,
        }
        result.update(reward)
        results.append(result)
    if save_path is not None:
        save_jsonl(results, save_path, overwrite=overwrite)


def eval_zero_shot_baseline(model: LLM, overwrite=False):
    path = project_dir.joinpath("data", "MATH", "validation.jsonl")
    data = pd.read_json(path_or_buf=path, lines=True)
    problems = list(data.problem)
    solutions = list(data.solution)
    evaluate_vllm(
        vllm_model=model,
        reward_fn=drgrpo_grader.r1_zero_reward_fn,
        prompts=problems,
        ground_truths=solutions,
        sampling_params=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        ),
        save_path="zero_baseline.jsonl",
        overwrite=overwrite,
    )


def load_zero_shot_baseline():
    path = project_dir.joinpath("evaluation", "zero_baseline.jsonl")
    data = pd.read_json(path_or_buf=path, lines=True)
    return data


def encode2(tokenizer: PreTrainedTokenizerBase):
    sft_data = load_sft()
    prompts = list(sft_data.prompt)
    responses = list(sft_data.response)
    return tokenize_prompt_and_output(make_prompt(prompts), responses, tokenizer)


# ----- train -----


def tokenize_prompt_and_output(
    prompt_strs, output_strs, tokenizer: PreTrainedTokenizerBase
):
    batch_encoding = tokenizer.batch_encode_plus(
        [(prompt_strs[i], output_strs[i]) for i in range(len(prompt_strs))],
        padding=True,
        return_token_type_ids=1,
        return_tensors="pt",
    )
    input_ids = batch_encoding["input_ids"]  # (batch_size, max_seq_len)
    labels = input_ids[:, 1:].clone()  # (batch_size, max_seq_len-1)
    input_ids = input_ids[:, :-1]  # (batch_size, max_seq_len-1)
    response_mask = batch_encoding["token_type_ids"].bool()[:, 1:]
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, sequence_length, vocab_size)
    entropy: (batch_size, sequence_length)
    """
    # m = logits.max(-1, keepdim=True)[0]
    # diff = logits - m
    # log_p = diff - torch.logsumexp(diff, dim=-1, keepdim=True)
    log_p = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    entropy = -(p * log_p).sum(-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1))[..., 0]
    result = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    tensor = torch.where(mask, tensor, 0) / normalize_constant
    tensor = tensor.sum(dim)
    return tensor


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    policy_log_probs (batch_size, sequence_length)
    response_mask (batch_size, sequence_length)
    """
    loss = -masked_normalize(
        policy_log_probs, response_mask, normalize_constant, dim=-1
    )  # (batch_size,)
    loss = loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}


@torch.inference_mode()
def log_generations(
    model: PreTrainedModel, tokensizer: PreTrainedTokenizerBase, validation_data
):
    prompt = tokensizer.encode(validation_data)
    model(prompt).logits


def init_wandb(name, **config):
    run = wandb.init(
        entity="ztzhu11",
        project="FT_Qwen2.5-Math-1.5B",
        name=name,
        config=config,
        # resume="must",
        # id="s95tyi64",
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step", summary="min")
    wandb.define_metric("eval/*", step_metric="eval_step", summary="min")
    return run


def init_vllm(device: str, seed: int, gpu_memory_utilization: float = 0.85):
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir):
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


def load_model(device=None):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    if device is not None:
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    return model, tokenizer


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    lr=1e-4,
    max_steps=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    log_name=None,
):
    if log_name is not None:
        run = init_wandb(
            name=log_name,
            lr=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        run = nullcontext()
    sft_data = load_sft()
    N = min(
        max_steps * per_device_train_batch_size * gradient_accumulation_steps,
        len(sft_data),
    )
    indexes = np.random.choice(len(sft_data), N, replace=False)
    stride = per_device_train_batch_size * gradient_accumulation_steps
    with run:
        checkpoint_step = 0
        for step in trange(max_steps):
            batch_indexes = indexes[step * stride : (step + 1) * stride]
            losses = []
            token_entropies = []
            for j in range(gradient_accumulation_steps):
                start = j * per_device_train_batch_size
                _batch_indexes = batch_indexes[
                    start : start + per_device_train_batch_size
                ]
                prompts = sft_data.iloc[_batch_indexes].prompt.tolist()
                responses = sft_data.iloc[_batch_indexes].response.tolist()
                input_data = tokenize_prompt_and_output(make_prompt(prompts), responses)
                response_mask = input_data["response_mask"].to(device)
                result = get_response_log_probs(
                    model,
                    input_data["input_ids"].to(device),
                    input_data["labels"].to(device),
                    return_token_entropy=True,
                )
                log_probs = result["log_probs"]
                normalize_constant = response_mask.sum(-1, keepdim=True)

                token_entropy = masked_normalize(
                    result["token_entropy"], response_mask, normalize_constant, dim=-1
                )  # (batch_size,)
                token_entropy = token_entropy.mean() / gradient_accumulation_steps
                token_entropies.append(token_entropy.detach().cpu().numpy().item())

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    normalize_constant=normalize_constant,
                )
                losses.append(loss.detach().cpu().numpy().item())
            loss = sum(losses)
            token_entropy = sum(token_entropies)
            step += 1
            checkpoint_step = step
            if step % 1 == 0:
                tqdm.write(
                    f"Step [{step}/{max_steps}], loss: {loss:.4f}, token_entropy: {token_entropy:.4f}"
                )
            run.log(
                {
                    "train_step": step,
                    "train/loss": loss,
                    "train/token_entropy": token_entropy,
                }
            )
        if log_name is not None:
            save_model(
                model,
                tokenizer,
                output_dir=project_dir
                / "checkpoints"
                / f"{log_name}_step{checkpoint_step}",
            )
            run.finish()
