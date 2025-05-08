# app/rag_agent.py
from config import OPENAI_API_KEY, PINECONE_API_KEY
from utils import parse_query_string, format_documents
from vectorstore import query_response
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, OpenAISettings
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from router_agent import LLMRouter
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


async def run_parallel_agent(query_string):
    app = MCPApp(name="rag_parallel", settings=settings)

    async with app.run():
        parsed = parse_query_string(query_string)
        response = query_response(parsed)
        formatted = format_documents(response)

        # Create the specialized agents
        rag_agent = Agent(
            name="rag_career_agent",
            instruction=f"""You are a helpful, encouraging career assistant for
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

        mentor_connect_agent = Agent(
            name="mentor_connect_agent",
            instruction=f"""You are a supportive career guide designed to
            connect students with mentors and career professionals. Your main
            job is to help students prepare for meaningful conversations with
            these individuals by:

            1. Offering conversation prompts they can use
            2. Letting them know about career professionals from our knowledge
            base that would be interested in talking to students

            Use the people recommended below to offer their experience to
            students and ask students if they'd like to interact with these
            people and offer them a template for reaching out to them.
            Knowledge:
            {formatted}""",
            server_names=[]
        )

        synthesizer = Agent(
            name="Synthesizer",
            instruction="""You are a synthesizer that combines responses from multiple agents:
            1) the RAG Career Agent, which draws from real interview transcripts
            with career professionals
            2) the Career Agent, which provides general career advice
            3) the Mentor Connect Agent, which helps connect students with mentors

            Your job is to produce a single, cohesive response that:
            - Clearly blends insights from all relevant agents
            - Attributes quotes or experiences from transcripts to specific individuals
            - Uses the general advice to provide context, structure, or broader takeaways
            - Includes mentor connection opportunities when relevant
            - Maintains a friendly, encouraging tone suitable for students exploring careers""",
            server_names=[]
        )

        # Initialize the router
        llm = OpenAIAugmentedLLM()
        router = LLMRouter(
            llm=llm,
            agents=[rag_agent, general_agent, mentor_connect_agent]
        )

        # Route the request to appropriate agents
        task = f"""A student will be messaging you about inquiries about career
        questions and guidance. You'll have a rag_career_agent that can answer
        specific questions based off people's experience, a career_agent that
        can answer general questions, and a mentor_connect_agent that will help
        students connect with mentors and professionals when needed.

        Student message: {parsed['content_string_query']}

        Answer:
        """

        # Get top 2 most relevant agents
        routed_agents = await router.route_to_agent(request=task, top_k=2)
        selected_agents = [agent for agent, _ in routed_agents]

        # Use parallel processing to get responses from selected agents
        parallel = ParallelLLM(
            fan_in_agent=synthesizer,
            fan_out_agents=selected_agents,
            llm_factory=OpenAIAugmentedLLM,
        )

        result = await parallel.generate_str(
            message=f"""
            Question: {parsed['content_string_query']}
            Answer:
            """
        )

        return result
