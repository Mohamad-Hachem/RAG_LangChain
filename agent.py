from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from prompts import RAG_WORKFLOW_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS, CHUNK_ANALYST_INSTRUCTIONS
from tools import search_documentation, backend
from langchain.messages import HumanMessage

max_concurrent_analysts = 3

INSTUCTUTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "="*80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "analyse one retreived documentation chunk file"
        "Pass the use question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS
}

model = init_chat_model(model="openai:gpt-5.5")

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTUCTUTIONS,
    subagents=[chunk_analyst_subagent]
)

example_query = "what are the cause of climate change"

result = agent.invoke(
    {"messages": [HumanMessage(content=example_query)]}
)

print(result["messages"][-1].content)