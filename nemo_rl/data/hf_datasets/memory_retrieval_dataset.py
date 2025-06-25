from typing import Any

import os
from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def _format(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": example["question"]}],
        "answer": example["answer"],
        "static_memory": example.get("static_memory", ""),
        "task_name": "memory_retrieval",
    }


class MemoryRetrievalDataset:
    """Load a JSON dataset for the memory retrieval task."""

    def __init__(self, path: str, val_split_ratio: float | None = 0.1) -> None:
        ds = load_dataset("json", data_files=path)["train"]

        if val_split_ratio is not None and val_split_ratio > 0:
            split = ds.train_test_split(test_size=val_split_ratio, seed=42)
            train_ds = split["train"].map(_format, remove_columns=ds.column_names)
            val_ds = split["test"].map(_format, remove_columns=ds.column_names)
        else:
            train_ds = ds.map(_format, remove_columns=ds.column_names)
            val_ds = None

        self.formatted_ds = {"train": train_ds, "validation": val_ds}
        
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to the repo root (assuming nemo_rl/data/hf_datasets/ structure)
        repo_root = os.path.join(current_dir, "..", "..", "..")
        # Construct the absolute path to the system prompt
        system_prompt_path = os.path.join(repo_root, "obsidian_agent", "agent", "system_prompt.txt")
        
        self.task_spec = TaskDataSpec(
            task_name="memory_retrieval",
            system_prompt_file=system_prompt_path,
        )
