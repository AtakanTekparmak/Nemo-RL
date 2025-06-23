import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Optional, cast

import torch
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

from nemo_rl.data.hf_datasets.memory_retrieval_dataset import MemoryRetrievalDataset
from nemo_rl.environments.memory_retrieval_environment import MemoryRetrievalEnvironment

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

TokenizerType = PreTrainedTokenizerBase


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GRPO training for memory retrieval")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()
    return args, overrides


def memory_retrieval_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    question = datum_dict["messages"][0]["content"]
    extra_env_info = {
        "answer": datum_dict["answer"],
        "static_memory": datum_dict.get("static_memory", ""),
    }

    message_log: LLMMessageLogType = []

    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(sys, return_tensors="pt")["input_ids"][0]
        message_log.append(sys_prompt)

    if task_data_spec.prompt:
        question = task_data_spec.prompt.format(question)
    user_message = {"role": "user", "content": question}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length > max_seq_length:
        for m in message_log:
            m["token_ids"] = m["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "memory_retrieval"),
    }
    return output


def setup_data(
    tokenizer: TokenizerType,
    data_config: dict,
    env_configs: dict[str, Any],
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data...")
    dataset_obj = MemoryRetrievalDataset(data_config["dataset_path"])
    task_spec = dataset_obj.task_spec

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = {
        "memory_retrieval": (task_spec, memory_retrieval_processor)
    }

    env = MemoryRetrievalEnvironment.options(  # type: ignore
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.memory_retrieval_environment.MemoryRetrievalEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["memory_retrieval"])

    dataset = AllTaskProcessedDataset(
        dataset_obj.formatted_ds["train"],
        tokenizer,
        task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: env)
    task_to_env["memory_retrieval"] = env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_memory_retrieval.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(tokenizer, config["data"], config["env"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
