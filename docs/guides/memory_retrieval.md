# Training the Obsidian Memory Agent with NeMo RL

This guide explains how to train the Obsidian memory agent on a retrieval task using NeMo RL's GRPO algorithm.

## Dataset Format
The training dataset should be a JSON list where each element contains the question, the expected answer and a snapshot of the agent's memory.

```json
[
  {
    "question": "What is my age?",
    "answer": "age: 34",
    "task": "retrieval",
    "static_memory": "{\"guideline\":\"...\", ...}",
    "persona": "Elena Müller",
    "fact": "age: 34"
  }
]
```

`static_memory` is the serialized content of the agent's memory directory that will be instantiated before each episode.

## Running Training
1. Place your dataset JSON somewhere accessible.
2. Edit `examples/configs/grpo_memory_retrieval.yaml` and set `data.dataset_path` to the location of the dataset.
3. Launch training with `uv`:

```bash
uv run examples/run_memory_retrieval.py --config examples/configs/grpo_memory_retrieval.yaml
```

The script loads the `Qwen/Qwen3-4B` model and trains it with GRPO. Checkpoints will be written to `results/grpo-memory` and logs to `logs/`.
