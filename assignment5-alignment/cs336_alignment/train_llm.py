import json
from pathlib import Path
import sys
from typing import Callable, List, Union

from vllm import LLM, SamplingParams
import pandas as pd

from cs336_alignment import drgrpo_grader

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

with open(project_dir / "cs336_alignment" / "prompts" / "r1_zero.prompt", "r") as f:
    r1_zero_prompt = f.read()


def make_prompt(
    input_text: Union[str, List[str]], prompt_type="r1_zero"
) -> Union[str, List[str]]:
    if not isinstance(input_text, str):
        return [make_prompt(t, prompt_type=prompt_type) for t in input_text]

    if prompt_type == "r1_zero":
        return r1_zero_prompt.format(question=input_text)
    else:
        raise NotImplementedError()


def save_jsonl(lines, path, overwrite=False):
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
    eval_sampling_params: SamplingParams,
    save_path=None,
    overwrite=False,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    eval_sampling_params = SamplingParams(
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
    outputs = vllm_model.generate(prompts, eval_sampling_params)
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
        eval_sampling_params=SamplingParams(
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
