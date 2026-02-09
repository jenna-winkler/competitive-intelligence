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
from agentstack_sdk.a2a.extensions import (
    CitationExtensionServer, CitationExtensionSpec,
    TrajectoryExtensionServer, TrajectoryExtensionSpec,
)
from agentstack_sdk.a2a.types import AgentMessage

from .streaming_citation_parser import StreamingCitationParser

server = Server()


def format_trajectory_content(tool_name: str, tool_input, tool_output) -> tuple[str, str]:
    """Format trajectory content for clean demo display"""
    
    if tool_name == "think":
        thoughts = tool_input.get("thoughts", "") if isinstance(tool_input, dict) else str(tool_input)
        return "🤔 Planning", f"Strategy: {thoughts}"
    
    elif tool_name == "duckduckgo_search":
        query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
        
        # Parse the output to count sources
        if isinstance(tool_output, str):
            # Try to parse as list of results
            import json
            try:
                results = json.loads(tool_output)
                if isinstance(results, list):
                    source_count = len(results)
                    # Show first 3 titles as preview
                    preview_titles = [r.get('title', '') for r in results[:3] if isinstance(r, dict)]
                    preview = '\n'.join(f"- {title}" for title in preview_titles if title)
                    if source_count > 3:
                        preview += f"\n- ... and {source_count - 3} more"
                    
                    output_summary = f"**Found {source_count} sources**\n\n{preview}"
                else:
                    output_summary = f"Output: {str(tool_output)[:200]}..."
            except:
                output_summary = f"Output: {str(tool_output)[:200]}..."
        else:
            output_summary = f"Output: {str(tool_output)[:200]}..."
        
        return f"🛠️ DuckDuckGo", f"Input: {{'query': '{query}'}}\n\n{output_summary}"
    
    else:
        input_str = str(tool_input)[:100]
        output_str = str(tool_output)[:200]
        return f"🛠️ {tool_name}", f"Input: {input_str}...\n\nOutput: {output_str}..."


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
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
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
        instructions=(
            "You are a competitive intelligence analyst. Research companies, analyze market positioning, "
            "identify trends, and provide strategic insights. Always start by thinking through what information "
            "would be most valuable, then gather current data to support your analysis. "
            "For search results, ALWAYS use proper markdown citations: [description](URL). "
            "Examples: [OpenAI releases GPT-5](https://example.com/gpt5), "
            "[AI adoption increases 67%](https://example.com/ai-study)"
        ),
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1),
            ConditionalRequirement(DuckDuckGoSearchTool, only_after=[ThinkTool], min_invocations=2, max_invocations=3),
        ],
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    response_text = ""
    citation_parser = StreamingCitationParser()

    async for event, meta in agent.run(
        user_message,
        expected_output="Markdown format with proper [text](URL) citations for search results.",
    ):
        match event:
            case RequirementAgentFinalAnswerEvent(delta=delta):
                response_text += delta
                clean_text, new_citations = citation_parser.process_chunk(delta)
                if clean_text:
                    yield clean_text
                if new_citations:
                    yield citation.citation_metadata(citations=new_citations)
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

    if final_text := citation_parser.finalize():
        yield final_text

    await context.store(AgentMessage(
        text=response_text,
        metadata=(citation.citation_metadata(citations=citation_parser.citations) if citation_parser.citations else None),
    ))


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