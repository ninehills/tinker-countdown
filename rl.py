import asyncio
import logging
from datetime import datetime
from functools import partial
from typing import Sequence

import chz
from datasets import load_dataset
from tinker.types import LossFnType
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from eval import (
    build_prompt_messages,
    check_format_tags,
    check_solution,
    extract_reasoning_and_solution,
)

logger = logging.getLogger(__name__)


class CountdownEnv(ProblemEnv):
    """Simple countdown environment."""

    def __init__(
        self,
        nums: list[int],
        target: int,
        renderer: renderers.Renderer,
    ):
        self.nums = nums
        self.target = target
        self.renderer = renderer
        self.prompt = build_prompt_messages(nums, target)
        convo_prefix = [self.prompt[0]]
        super().__init__(renderer, convo_prefix, format_coef=0.1)

    def get_question(self) -> str:
        return self.prompt[1]["content"]

    def check_answer(self, sample_str: str) -> bool:
        _, solution = extract_reasoning_and_solution(sample_str)
        is_correct, _ = check_solution(self.nums, self.target, solution)
        return is_correct

    def check_format(self, sample_str: str) -> bool:
        if not check_format_tags(sample_str):
            return False
        return True

    def get_reference_answer(self) -> str:
        # A valid answer: sum all numbers
        return " ".join(map(str, self.nums))


class CountdownDataset(RLDataset):
    def __init__(
        self,
        data: list[dict],
        batch_size: int,
        renderer: renderers.Renderer,
        group_size: int,
        n_batches: int = 100
    ):
        self.data = data
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.n_batches = n_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # Cycle through dataset deterministically for reproducibility
        return [
            self._make_env_group_builder(self.data[(index * self.batch_size + i) % len(self.data)])
            for i in range(self.batch_size)
        ]

    def _make_env_group_builder(self, datum: dict) -> ProblemGroupBuilder:
        nums = datum["nums"]
        target = datum["target"]
        return ProblemGroupBuilder(
            env_thunk=partial(
                CountdownEnv, nums, target, renderer=self.renderer
            ),
            num_envs=self.group_size,
        )

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class CountdownDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_batches: int
    group_size: int
    limit: int

    async def __call__(self) -> tuple[CountdownDataset, CountdownDataset]:
        train_ds = (
            load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "all")["train"]
            .filter(lambda x: len(x["nums"]) == 4)
            .shuffle(seed=42)
            .select(range(self.limit))
            .select_columns(["nums", "target"])
            .to_list()
        )
        eval_ds = (
            load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "test")["test"]
            .filter(lambda x: len(x["nums"]) == 4)
            .shuffle(seed=42)
            .select(range(200))
            .select_columns(["nums", "target"])
            .to_list()
        )
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            CountdownDataset(
                data=train_ds,
                batch_size=self.batch_size,
                renderer=renderer,
                n_batches=self.n_batches,
                group_size=self.group_size,
            ),
            CountdownDataset(
                data=eval_ds,
                batch_size=200,
                renderer=renderer,
                n_batches=1,
                group_size=1,
            ),
        )


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    lora_rank: int = 16
    group_size: int = 4
    groups_per_batch: int = 64
    limit: int = 10000
    learning_rate: float = 5e-5
    max_tokens: int = 3072
    temperature: float = 1.0
    n_batches: int = 200
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    eval_every: int = 10
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    load_checkpoint_path: str | None = None
    compute_post_kl: bool = False
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps_off_policy: int | None = None
    loss_fn: LossFnType = "importance_sampling"


def make_config(cli_config: CLIConfig) -> Config:
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_slug = cli_config.model_name.replace("/", "-")
    run_name = f"countdown-{model_slug}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-limit{cli_config.limit}"
    log_path = cli_config.log_path or f"logs/countdown_rl/{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    dataset_builder = CountdownDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        n_batches=cli_config.n_batches,
        group_size=cli_config.group_size,
        limit=cli_config.limit,
    )

    async_config = (
        AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None
    )

    return Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=async_config,
        loss_fn=cli_config.loss_fn,
    )


async def cli_main(cli_config: CLIConfig):
    config = make_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
