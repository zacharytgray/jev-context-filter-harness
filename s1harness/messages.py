# the fixed messages. the wording is measured, don't edit casually. the filters' lines are in pipeline.py

AGENT_PROMPT = "You are an agent working in a repository. Use the tools to do what the user asks."
FINAL_PROMPT = """When the work is finished, or you must stop, reply without a tool call. That reply is your final message.
Say what is done, what is not, and why the run is ending."""


def system_prompt() -> str:
    return AGENT_PROMPT + "\n\n" + FINAL_PROMPT
