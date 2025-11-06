from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Callable, List, Union
from unittest.mock import patch

import numpy as np
from openai import OpenAI
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed
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


def load_sft(max_samples=None):
    path = project_dir.joinpath("data", "MATH", "sft.jsonl")
    sft_data = pd.read_json(
        path_or_buf=path,
        lines=True,
        dtype={"data_type": str, "data_index": int, "prompt": str, "response": str},
    )
    if max_samples is not None:
        sft_data = sft_data.iloc[:max_samples]
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
                {
                    "role": "user",
                    "content": pre_prompt + data.iloc[index].to_json(force_ascii=False),
                },
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
    # sft_data.to_json(path, lines=True, orient="records", force_ascii=False)
    save_jsonl(sft_data.to_dict(orient="records"), path, overwrite=True)


def check_sft_dataset(fast=False):
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
            sft_data.iloc[i].response, data.iloc[index].solution, fast=fast
        )
        if reward["reward"] != 1:
            rewards[index] = reward
    return rewards


def save_jsonl(lines: Union[list[dict], pd.DataFrame], path, overwrite=False):
    if isinstance(lines, pd.DataFrame):
        lines = lines.to_dict(orient="records")

    path = Path(path)
    if not overwrite:
        assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, line in enumerate(lines):
            json_line = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
            if i == 0:
                f.write(json_line)
            else:
                f.write("\n" + json_line)
    print(f"Saved {len(lines)} lines to {path.as_posix()}")


def load_jsonl(path):
    data = pd.read_json(path_or_buf=path, lines=True)
    return data


def evaluate_vllm(
    llm: LLM,
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
    outputs = llm.generate(prompts, sampling_params)

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
    return results


def evaluate_pretrained(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    generation_config: GenerationConfig,
    batch_size=8,
    save_path=None,
    overwrite=False,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        min_new_tokens=4,
        max_new_tokens=1024,
        stop_strings=["</answer>"],
        do_sample=True,
    )
    """
    if save_path is not None:
        save_path = project_dir / "evaluation" / save_path
        if not overwrite:
            assert not save_path.exists()
    raw_prompts = prompts
    prompts = make_prompt(prompts)
    tokenizer.padding_side = "left"

    outputs = []
    bar = tqdm(total=len(prompts))
    input_tokens = 0
    output_tokens = 0
    t0 = time.time()
    for i in range(np.ceil(len(prompts) / batch_size).astype(int)):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        input_tokens += inputs["attention_mask"].sum()
        _outputs = model.generate(
            **inputs, generation_config=generation_config, tokenizer=tokenizer
        )
        _outputs = _outputs[:, inputs["input_ids"].shape[1] :]
        _outputs = tokenizer.batch_decode(_outputs, skip_special_tokens=True)
        for i in range(len(_outputs)):
            index = _outputs[i].find("</answer>")
            if index >= 0:
                _outputs[i] = _outputs[i][: index + len("</answer>")]
        outputs.extend(_outputs)
        for o in _outputs:
            output_tokens += len(tokenizer.encode(o))
        t = time.time() - t0
        bar.update(len(_outputs))
        bar.set_postfix_str(
            f"input: {input_tokens/t:.1f} toks/s, output: {output_tokens/t:.1f} toks/s"
        )
    bar.close()

    results = []
    for i, output in enumerate(outputs):
        response = output
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
    return results


def evaluate_dataset(
    model: Union[LLM, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase = None,
    batch_size=8,
    data_name="validation",
    prompts=None,
    ground_truths=None,
    save_path=None,
    overwrite=False,
):
    if prompts is None:
        assert ground_truths is None
        path = project_dir.joinpath("data", "MATH", f"{data_name}.jsonl")
        data = load_jsonl(path)
        prompts = list(data.problem)
        ground_truths = list(data.solution)
    if isinstance(model, LLM):
        return evaluate_vllm(
            llm=model,
            reward_fn=drgrpo_grader.r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            sampling_params=SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=1024,
                stop=["</answer>"],
                include_stop_str_in_output=True,
            ),
            save_path=save_path,
            overwrite=overwrite,
            # max_num_seqs=batch_size,
        )
    else:
        pad_token = tokenizer.special_tokens_map["pad_token"]
        return evaluate_pretrained(
            model=model,
            tokenizer=tokenizer,
            reward_fn=drgrpo_grader.r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            generation_config=GenerationConfig(
                temperature=1.0,
                top_p=1.0,
                min_new_tokens=4,
                max_new_tokens=1024,
                stop_strings=["</answer>"],
                do_sample=True,
                pad_token_id=tokenizer.encode(pad_token)[0],
            ),
            batch_size=batch_size,
            save_path=save_path,
            overwrite=overwrite,
        )


def load_zero_shot_baseline():
    path = project_dir.joinpath("evaluation", "zero_baseline.jsonl")
    data = pd.read_json(path_or_buf=path, lines=True)
    return data


# ----- train -----


def tokenize_prompt_and_output(
    prompt_strs, output_strs, tokenizer: PreTrainedTokenizerBase, device=None
):
    tokenizer.padding_side = "right"
    batch_encoding = tokenizer(
        [(prompt_strs[i], output_strs[i]) for i in range(len(prompt_strs))],
        padding=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    input_ids = batch_encoding["input_ids"]  # (batch_size, max_seq_len)
    labels = input_ids[:, 1:].clone()  # (batch_size, max_seq_len-1)
    input_ids = input_ids[:, :-1]  # (batch_size, max_seq_len-1)
    response_mask = batch_encoding["token_type_ids"].bool()[:, 1:]
    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        response_mask = response_mask.to(device)
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


def init_vllm(seed: int = 3407, gpu_memory_utilization: float = 0.85):
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
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


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir,
    overwrite=False,
):
    if not overwrite:
        assert not Path(output_dir).exists()
    model.save_pretrained(save_directory=output_dir, max_shard_size="10GB")
    tokenizer.save_pretrained(save_directory=output_dir)


def load_model(
    model_path="Qwen/Qwen2.5-Math-1.5B",
    device=None,
    dtype=torch.float16,
    padding_side="right",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
    )
    if device is not None:
        model.to(device)
    tokenizer = load_tokenizer(model_path=model_path, padding_side=padding_side)
    return model, tokenizer


def load_tokenizer(model_path="Qwen/Qwen2.5-Math-1.5B", padding_side="right"):
    """
    for decoder-only models, padding_side should be "right" for training, "left" for inference
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
    return tokenizer


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    lr=1e-4,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    eval_per_epoch=True,
    max_steps=None,
    max_samples=None,
    sft_data=None,
    log_name=None,
    save=True,
):
    betas = (0.9, 0.999)
    weight_decay = 0.0
    max_grad_norm = 1.0

    if sft_data is None:
        sft_data = load_sft(max_samples=max_samples)
    if max_samples is None:
        max_samples = len(sft_data)
    per_step_train_batch_size = (
        per_device_train_batch_size * gradient_accumulation_steps
    )  # assume num_devices = 1
    step_per_epoch = len(sft_data) // per_step_train_batch_size
    assert step_per_epoch > 0, "Not enough data for training."
    if max_steps is None:
        max_steps = step_per_epoch * num_train_epochs
    num_train_steps = min(step_per_epoch * num_train_epochs, max_steps)
    indexes = None
    if log_name is not None:
        run = init_wandb(
            name=log_name,
            lr=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_step_train_batch_size=per_step_train_batch_size,
            betas=betas,
            weight_decay=weight_decay,
            max_samples=max_samples,
            num_train_epochs=num_train_epochs,
            num_train_steps=num_train_steps,
        )
    else:
        run = nullcontext()
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr / 10)
    optimizer.zero_grad()
    with run:
        checkpoint_step = 0
        checkpoint_epoch = 0
        checkpoint_sample = 0
        t0 = time.time()
        for step in trange(num_train_steps):
            epoch = step // step_per_epoch
            is_new_epoch = step % step_per_epoch == 0
            if is_new_epoch:
                indexes = np.random.choice(
                    len(sft_data),
                    per_step_train_batch_size * step_per_epoch,
                    replace=False,
                )

            batch_indexes = indexes[
                (step % step_per_epoch)
                * per_step_train_batch_size : (step % step_per_epoch + 1)
                * per_step_train_batch_size
            ]
            losses = []
            token_entropies = []
            for j in range(gradient_accumulation_steps):
                start = j * per_device_train_batch_size
                _batch_indexes = batch_indexes[
                    start : start + per_device_train_batch_size
                ]
                prompts = sft_data.iloc[_batch_indexes].prompt.tolist()
                responses = sft_data.iloc[_batch_indexes].response.tolist()
                input_data = tokenize_prompt_and_output(
                    make_prompt(prompts), responses, tokenizer, device=device
                )
                input_ids = input_data.pop("input_ids")
                labels = input_data.pop("labels")
                response_mask = input_data.pop("response_mask")
                result = get_response_log_probs(
                    model,
                    input_ids,
                    labels,
                    return_token_entropy=True,
                )
                log_probs = result["log_probs"]
                normalize_constant = response_mask.sum(-1, keepdim=True)

                with torch.inference_mode():
                    token_entropy = masked_normalize(
                        result["token_entropy"],
                        response_mask,
                        normalize_constant,
                        dim=-1,
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

                # del input_ids, labels
                # torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            epoch += 1
            checkpoint_epoch = epoch
            checkpoint_step = step
            checkpoint_sample += len(batch_indexes)
            model_name = f"{log_name}-sample{max_samples}-epoch{checkpoint_epoch}-step{checkpoint_step}"

            if step % step_per_epoch == 0 and eval_per_epoch:  # the epoch ends
                # evaluate_dataset(model, tokenizer, save_path=f"{model_name}.jsonl")
                if num_train_steps != step: # save to eval later
                    save_model(
                        model,
                        tokenizer,
                        output_dir=project_dir / "checkpoints" / model_name,
                    )

            # ----- log -----
            loss = sum(losses)
            token_entropy = sum(token_entropies)
            if step % 1 == 0:
                tqdm.write(
                    f"[{epoch}/{num_train_epochs}, {step}/{num_train_steps}], loss: {loss:.4f}, token_entropy: {token_entropy:.4f}"
                )
            if log_name is not None:
                run.log(
                    {
                        "train_step": step,
                        "train/epoch": epoch,
                        "train/loss": loss,
                        "train/token_entropy": token_entropy,
                        "train/time": time.time() - t0,
                    }
                )
        if log_name is not None:
            if save:
                save_model(
                    model,
                    tokenizer,
                    output_dir=project_dir / "checkpoints" / model_name,
                )
            run.finish()


def make_ei_dataset(
    model: Union[LLM, PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase = None,
    ei_batch_size=512,
    rollouts=2,
    seed=42,
    step=None,
):
    path = project_dir.joinpath("data", "MATH", "train.jsonl")
    data = pd.read_json(path_or_buf=path, lines=True)
    np.random.seed(seed)
    indexes = np.random.choice(len(data), ei_batch_size, replace=False)
    indexes = np.sort(indexes)

    ei_data = pd.DataFrame(columns=["data_type", "data_index", "prompt", "response"])
    problems = data.problem[indexes].tolist()
    solutions = data.solution[indexes].tolist()
    for _ in range(rollouts):
        results = evaluate_dataset(
            model,
            tokenizer,
            batch_size=8,
            prompts=problems,
            ground_truths=solutions,
        )

        for i, result in enumerate(results):
            if result["reward"] == 0:
                continue
            ei_data.loc[len(ei_data)] = {
                "data_type": "train",
                "data_index": indexes[i],
                "prompt": result["problem"],
                "response": result["response"],
            }
    success_rate = len(ei_data) / (ei_batch_size * rollouts)
    print(f"success_rate: {success_rate:.3f}")
    if len(ei_data) and step is not None:
        path = project_dir.joinpath(
            "data", "MATH", f"ei_{ei_batch_size}_step{step}.jsonl"
        )
        save_jsonl(ei_data.to_dict(orient="records"), path)
    return ei_data


def train_ei(n_ei_steps=5):
    model, tokenizer = load_model(device=device)
    ei_data = load_jsonl(project_dir.joinpath("data", "MATH", "ei_512_step1.jsonl"))
    for ei_step in range(n_ei_steps):
        if ei_step > 0:
            ei_data = make_ei_dataset(
                model,
                tokenizer,
                ei_batch_size=512,
                rollouts=2,
                seed=3407 + ei_step,
                step=ei_step + 1,
            )
        train(
            model,
            tokenizer,
            num_train_epochs=3,
            gradient_accumulation_steps=16,
            sft_data=ei_data,
            log_name=f"ei512-ei_step{ei_step+1}",
            save=ei_step == n_ei_steps - 1,
        )
