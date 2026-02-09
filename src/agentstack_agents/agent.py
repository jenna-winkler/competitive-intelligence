import os
from dotenv import load_dotenv
load_dotenv()

from openinference.instrumentation.beeai import BeeAIInstrumentor
BeeAIInstrumentor().instrument()

from typing import Annotated
from a2a.types import Message, AgentSkill, Role
from a2a.utils.message import get_message_text
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.requirement.events import (
    RequirementAgentFinalAnswerEvent,
    RequirementAgentSuccessEvent,
)
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend.message import UserMessage as FrameworkUserMessage, AssistantMessage as FrameworkAssistantMessage
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from agentstack_sdk.a2a.extensions import TrajectoryExtensionServer, TrajectoryExtensionSpec
from agentstack_sdk.a2a.types import AgentMessage

server = Server()


def format_trajectory_content(tool_name: str, tool_input, tool_output) -> tuple[str, str]:
    """Format trajectory content for clean demo display"""
    
    if tool_name == "think":
        thoughts = tool_input.get("thoughts", "") if isinstance(tool_input, dict) else str(tool_input)
        return "🤔 Planning", f"**Strategy:** {thoughts}"
    
    elif tool_name == "duckduckgo_search":
        query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
        return f"🔍 Searching: {query}", f"**Input:** {tool_input}\n\n**Output:** {tool_output}"
    
    else:
        return f"🛠️ {tool_name}", f"**Input:** {tool_input}\n\n**Output:** {tool_output}"


@server.agent(
    name="Competitive Intelligence Agent",
    skills=[
        AgentSkill(
            id="competitive-intelligence",
            name="Competitive Intelligence Research",
            description="Analyzes competitors, market trends, and strategic opportunities using real-time web search and strategic thinking.",
            tags=["research", "competitive-intelligence", "market-analysis"],
            examples=[
                "What emerging trends in agentic AI platforms should enterprise software teams be preparing for this year?",
                "Compare the product strategies of Anthropic, OpenAI, and Google in the AI agent space",
                "What are Microsoft's latest AI initiatives and how do they compete with AWS?",
            ]
        )
    ]
)
async def competitive_intel(
    input: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()]
):
    await context.store(input)

    history = [
        msg async for msg in context.load_history()
        if isinstance(msg, Message) and msg.parts
    ]

    memory = UnconstrainedMemory()
    for msg in history[:-1]:
        text = get_message_text(msg)
        if msg.role == Role.agent:
            await memory.add(FrameworkAssistantMessage(text))
        else:
            await memory.add(FrameworkUserMessage(text))

    user_message = get_message_text(input)

    agent = RequirementAgent(
        llm=ChatModel.from_name(
            os.getenv("MODEL", "openai:gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        tools=[ThinkTool(), DuckDuckGoSearchTool()],
        memory=memory,
        instructions="You are a competitive intelligence analyst. Research companies, analyze market positioning, identify trends, and provide strategic insights. Always start by thinking through what information would be most valuable, then gather current data to support your analysis.",
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1),
            ConditionalRequirement(DuckDuckGoSearchTool, only_after=[ThinkTool], min_invocations=2, max_invocations=3),
        ],
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    response_text = ""
    async for event, meta in agent.run(user_message):
        match event:
            case RequirementAgentFinalAnswerEvent(delta=delta):
                response_text += delta
                yield delta
            case RequirementAgentSuccessEvent(state=state):
                last_step = state.steps[-1]

                if last_step.tool.name == FinalAnswerTool.name:
                    continue

                title, content = format_trajectory_content(
                    last_step.tool.name,
                    last_step.input,
                    last_step.output
                )

                yield trajectory.trajectory_metadata(title=title, content=content)

    await context.store(AgentMessage(text=response_text))


def run():
    try:
        server.run(
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", 8000)),
            context_store=PlatformContextStore(),
            configure_telemetry=True,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()