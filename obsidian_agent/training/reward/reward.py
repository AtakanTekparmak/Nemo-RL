from obsidian_agent.training.reward.utils import construct_judge_prompt
from obsidian_agent.training.reward.model import get_model_response
from obsidian_agent.training.reward.schemas import JudgeResponse, Fact
from obsidian_agent.training.settings import GPT_O3

def get_reward(
        folder_dump_str: str,
        facts_to_check: list[Fact]
    ) -> float:
    """
    Get the LLM-as-a-judge reward for the given folder dump and facts to check.

    Args:
        folder_dump_str: The folder dump as a string.
        facts_to_check: The facts to check.

    Returns:
        The reward as a float. The reward is the ratio of the number of facts present in the folder dump to the total number of facts.
    """
    judge_prompt = construct_judge_prompt(folder_dump_str, facts_to_check)
    judge_response = get_model_response(
        schema=JudgeResponse,
        prompt=judge_prompt,
        model=GPT_O3
    )
    return judge_response.ratio_of_facts_present