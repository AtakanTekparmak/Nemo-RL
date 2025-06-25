install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh;
	uv venv;

install-agent: install-uv
	source .venv/bin/activate;
	uv pip install openai pydantic python-dotenv;

run-training: install-agent
	uv run run_memory_retrieval.py --config examples/configs/grpo_memory_retrieval.yaml;
