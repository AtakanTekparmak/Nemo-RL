from typing import Any

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

    def __init__(self, path: str):
        ds = load_dataset("json", data_files=path)["train"]
        formatted = ds.map(_format, remove_columns=ds.column_names)
        self.formatted_ds = {"train": formatted, "validation": None}
        self.task_spec = TaskDataSpec(
            task_name="memory_retrieval",
            system_prompt_file="obsidian-agent/agent/system_prompt.txt",
        )
