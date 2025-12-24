import asyncio
import os
from datetime import datetime
from typing import Any
from dataclasses import dataclass

import re
import json
import tinker
from tinker import types
from tqdm import tqdm
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


SYSTEM_PROMPT = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
USER_PROMPT_TPL = "Using the numbers [{nums}], create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
ALLOWED_PATTERN = re.compile(r"^[\d+\-*/().\s]+$")
THINK_ANSWER_PATTERN = re.compile(
    r"^(?!.*<think>.*<think>)(?!.*<answer>.*<answer>).*\<think\>.*?\</think\>.*\<answer\>.*?\</answer\>.*$",
    re.DOTALL,
)


def build_prompt_messages(nums: list[int], target: int) -> list[renderers.Message]:
    return [
        renderers.Message(role="system", content=SYSTEM_PROMPT),
        renderers.Message(
            role="user",
            content=USER_PROMPT_TPL.format(nums=', '.join(map(str, nums)), target=target),
        ),
    ]

def extract_reasoning_and_solution(content: str) -> tuple[str, str]:
    reasoning_str = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    reasoning = reasoning_str.group(1) if reasoning_str else ""
    answers = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
    solution = answers[-1].strip() if answers else ""
    return reasoning, solution


def check_format_tags(sample_str: str) -> bool:
    """
    Ensures exactly one <think></think> and one <answer></answer> pair in order.
    """
    return bool(THINK_ANSWER_PATTERN.match(sample_str))


def check_solution(nums: list[int], target: int, solution: str) -> tuple[bool, str]:
    if not solution:
        return False, "Solution is empty"
    try:
        normalized = solution.split("=")[0] if "=" in solution else solution
        if not ALLOWED_PATTERN.match(normalized):
            return False, "Solution is not allowed"
        result = eval(normalized, {"__builtins__": {}}, {})
        if result != target:
            return False, "Solution is not equal to target"
        used_numbers = [int(n) for n in re.findall(r"\d+", normalized)]
        if sorted(used_numbers) != sorted(nums):
            return False, "Solution is not using all numbers or each number is not used only once"
        return True, "Solution is correct"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"Solution calculation failed: {e}"


@dataclass
class EvalResult:
    accuracy: float
    total: int
    correct: int
    details: list[dict[str, Any]]


class CountdownEvaluator(SamplingClientEvaluator):
    """
    A toy SamplingClientEvaluator that runs a custom evaluation and returns its metrics.
    """

    def __init__(
        self,
        dataset: Any,
        model_name: str,
        renderer_name: str,
        concurrency: int = 10,
    ):
        self.dataset = dataset
        self.concurrency = concurrency

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> EvalResult:
        results = EvalResult(
            accuracy=0,
            total=len(self.dataset),
            correct=0,
            details=[],
        )
        sampling_params = types.SamplingParams(
            max_tokens=3072,
            temperature=0.6,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_one(datum: dict[str, Any]) -> dict[str, Any]:
            prompt = build_prompt_messages(datum["nums"], datum["target"])
            try:
                async with semaphore:
                    model_input: types.ModelInput = self.renderer.build_generation_prompt(prompt)
                    r: types.SampleResponse = await sampling_client.sample_async(
                        prompt=model_input, num_samples=1, sampling_params=sampling_params
                    )
                    tokens: list[int] = r.sequences[0].tokens
                    response: renderers.Message = self.renderer.parse_response(tokens)[0]
                    content = renderers.ensure_text(response["content"])
                    reasoning, solution = extract_reasoning_and_solution(content)
                    is_correct, reason = check_solution(datum["nums"], datum["target"], solution)
                    return {
                        "input": datum,
                        "completion": content,
                        "reasoning": reasoning,
                        "solution": solution,
                        "is_correct": is_correct,
                        "reason": reason,
                        "prompt": prompt,
                    }
            except Exception as e:
                return {
                    "input": datum,
                    "completion": "",
                    "reasoning": "",
                    "solution": "",
                    "is_correct": False,
                    "reason": f"infer error: {e}",
                    "prompt": prompt,
                }

        tasks = [asyncio.create_task(evaluate_one(datum)) for datum in self.dataset]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating", unit="sample"):
            detail = await task
            results.details.append(detail)
            results.correct += 1 if detail["is_correct"] else 0

        results.accuracy = results.correct / results.total
        return results


async def main(model: str, renderer: str, limit: int, concurrency: int):
    from datasets import load_dataset
    from dotenv import load_dotenv
    load_dotenv()

    # Load the test set for 4 numbers
    countdown_ds = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "test")["test"]\
        .filter(lambda x: len(x["nums"]) == 4)\
        .shuffle(seed=42).select(range(limit))\
        .select_columns(["nums", "target"])
    print("==== Countdown Eval====")
    print(">>> Datasets:")
    print(countdown_ds)

    # Load the model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=model
    )
    print(f">>> Model: {model}")

    # Load the evaluator
    evaluator = CountdownEvaluator(
        dataset=countdown_ds,
        renderer_name=renderer,
        model_name=model,
        concurrency=concurrency,
    )
    
    # Run the evaluation
    result = await evaluator(sampling_client)
    print(">>> Evaluation Result:")
    print(f"Accuracy: {result.accuracy} ({result.correct}/{result.total})")
    # 保存评测输出，使用模型名称作为目录
    os.makedirs("outputs", exist_ok=True)
    # Replace any characters that are not filename-friendly with underscores
    model_dir = re.sub(r"[^\w.-]+", "_", model)
    output_dir = os.path.join("outputs", model_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "renderer": renderer,
                "accuracy": result.accuracy,
                "total": result.total,
                "correct": result.correct,
                "details": result.details,
            },
            f,
            indent=2,
        )
    print(f">>> Saved evaluation output to {output_path}")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    # https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers.py
    parser.add_argument("--renderer", type=str, default="qwen3")
    parser.add_argument("--limit", type=int, default=100, help="Limit the number of examples to evaluate")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency of sampling")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.renderer, args.limit, args.concurrency))
