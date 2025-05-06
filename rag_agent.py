# app/rag_agent.py
from config import OPENAI_API_KEY, PINECONE_API_KEY
from utils import parse_query_string, format_documents
from vectorstore import query_response
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, OpenAISettings
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from pinecone import Pinecone

settings = Settings(
    mcp=MCPSettings(
        servers={
            "fetch": MCPServerSettings(name="fetch", command="uvx", args=["mcp-server-fetch"]),
            "filesystem": MCPServerSettings(name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem"])
        }
    ),
    openai=OpenAISettings(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
)

app = MCPApp(name="rag_agent", settings=settings)

async def run_career_agent():
    async with app.run():
        rag_career_agent = Agent(
            name="rag_career_agent",
            instruction="""You are a helpful, encouraging career assistant for students exploring future job paths. Only use transcripts.""",
            server_names=[],
        )

        query = '''{
            "content_string_query": "How does a journalist get into the industry?",
            "industry_filter": [],
            "takeaways_filter": []
        }'''

        parsed = parse_query_string(query)
        response = query_response(parsed)
        formatted = format_documents(response)

        llm = await rag_career_agent.attach_llm(OpenAIAugmentedLLM)
        result = await llm.generate_str(
            message=f"""
                Context:
                {formatted}
                Question: {parsed['content_string_query']}
                Answer:
            """
        )

        print("\nCareer Agent Result:\n", result)


async def run_parallel_agent():
    app = MCPApp(name="rag_parallel", settings=settings)

    async with app.run():
        query = '''{
            "content_string_query": "How does a journalist get into the industry?",
            "industry_filter": [],
            "takeaways_filter": []
        }'''

        parsed = parse_query_string(query)
        response = query_response(parsed)
        formatted = format_documents(response)

        rag_agent = Agent(
            name="rag_career_agent",
            instruction=f"""OYou are a helpful, encouraging career assistant for
            students exploring future job paths. You have a structured knowledge
            base of career transcripts that contain interviews of professionals
            in different careers that can answer questions based off their
            experience. Only use these transcripts to answer questions.
            Knowledge:
            {formatted}""",
            server_names=[]
        )
        general_agent = Agent(
            name="career_agent",
            instruction="""You are a helpful, encouraging career assistant for
            students exploring future job paths. Your role is to guide students
            in thinking critically about their interests, strengths, and goals
            by asking thoughtful questions, reflecting their input, and helping
            them explore career directions in a conversational, supportive way.""",
            server_names=[]
        )
        synthesizer = Agent(
            name="Synthesizer",
            instruction="""Synthesize the answers from the Rag Career Agent and
            the Career Agent into a cohesive answer that blends personal answers
            from career professionals and answers that can be more general. If
            a person has said that answer, say that that person said it.""",
            server_names=[]
        )

        parallel = ParallelLLM(
            fan_in_agent=synthesizer,
            fan_out_agents=[rag_agent, general_agent],
            llm_factory=OpenAIAugmentedLLM,
        )

        result = await parallel.generate_str(
            message=f"""
                Question: {parsed['content_string_query']}
                Answer:
            """
        )

        print("\nParallel Agent Result:\n", result)
