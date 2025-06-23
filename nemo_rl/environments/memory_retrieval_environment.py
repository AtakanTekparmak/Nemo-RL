import json
import os
import tempfile
from typing import Any, Optional, TypedDict

import ray
import torch

from obsidian_agent.agent.engine import execute_sandboxed_code
from obsidian_agent.agent.settings import MAX_TOOL_TURNS, SANDBOX_TIMEOUT
from obsidian_agent.agent.utils import extract_python_code, extract_reply
from obsidian_agent.training.reward.schemas import Fact
from obsidian_agent.training.reward import get_reward

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MemoryRetrievalEnvConfig(TypedDict, total=False):
    max_turns: int


class MemoryRetrievalMetadata(TypedDict, total=False):
    answer: str
    static_memory: str
    num_turns: int
    memory_dir: str
    _tmpdir: Any


@ray.remote
class MemoryRetrievalEnvironment(EnvironmentInterface):
    """Multi-turn environment for the Obsidian memory agent."""

    def __init__(self, cfg: Optional[MemoryRetrievalEnvConfig] = None):
        cfg = cfg or {}
        self.max_turns = cfg.get("max_turns", MAX_TOOL_TURNS)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _setup_memory(self, meta: MemoryRetrievalMetadata) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        memory_dir = tmpdir.name
        meta["_tmpdir"] = tmpdir
        meta["memory_dir"] = memory_dir
        try:
            data = json.loads(meta.get("static_memory", "{}"))
            guideline = data.get("guideline")
            if guideline:
                os.makedirs(memory_dir, exist_ok=True)
                with open(os.path.join(memory_dir, "guideline.md"), "w") as f:
                    f.write(guideline)
            user_path = data.get("user_file_path")
            user_content = data.get("user_file_content", "")
            if user_path:
                full = os.path.join(memory_dir, user_path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w") as f:
                    f.write(user_content)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata_batch: list[MemoryRetrievalMetadata],
    ) -> EnvironmentReturn:
        observations: list[dict[str, str]] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        next_stop_strings: list[Optional[list[str]] | None] = []
        next_metadata: list[Optional[MemoryRetrievalMetadata]] = []

        for message_log, meta in zip(message_log_batch, metadata_batch):
            meta = dict(meta) if meta is not None else {}
            if "memory_dir" not in meta:
                self._setup_memory(meta)
                meta["num_turns"] = 0
            memory_dir = meta["memory_dir"]
            num_turns = meta.get("num_turns", 0)

            # find last assistant message
            last = next((m["content"] for m in reversed(message_log) if m["role"] == "assistant"), "")
            python_code = extract_python_code(last)
            obs_content = ""
            reward = 0.0
            terminated = False
            new_meta: Optional[MemoryRetrievalMetadata] = meta.copy()

            if python_code:
                locals_dict, error = execute_sandboxed_code(
                    python_code,
                    timeout=SANDBOX_TIMEOUT,
                    allowed_path=memory_dir,
                    import_module="agent.tools",
                )
                result = error if error else locals_dict
                obs_content = f"<result>{result}</result>"
            else:
                obs_content = ""

            if "<reply>" in last and "</reply>" in last:
                reply_text = extract_reply(last)
                fact = Fact(fact_description=meta.get("answer", ""))
                reward = get_reward(folder_dump_str=reply_text, facts_to_check=[fact])
                terminated = True
                if "_tmpdir" in meta:
                    meta["_tmpdir"].cleanup()
                new_meta = None
            elif num_turns + 1 >= self.max_turns:
                terminated = True
                if "_tmpdir" in meta:
                    meta["_tmpdir"].cleanup()
                new_meta = None
            else:
                new_meta["num_turns"] = num_turns + 1

            observations.append({"role": "environment", "content": obs_content})
            rewards.append(reward)
            terminateds.append(terminated)
            next_stop_strings.append(None)
            next_metadata.append(new_meta)

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get("total_reward", torch.tensor([0.0] * len(batch["idx"])))
        success_rate = (final_rewards > 0).float().mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"memory_retrieval_success_rate": success_rate}
