install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh;
	uv venv;

install-agent: install-uv
	uv pip install openai pydantic python-dotenv;

run-training: 
	uv run run_memory_retrieval.py --config examples/configs/grpo_memory_retrieval.yaml;
