install-agent:
	uv pip install openai pydantic python-dotenv

run-training: install-agent
	uv run run_memory_retrieval.py --config examples/configs/grpo_memory_retrieval.yaml
