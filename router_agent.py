from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.agents.agent import Agent
from typing import List, Tuple
import json

class LLMRouter:
    def __init__(self, llm: OpenAIAugmentedLLM, agents: List[Agent]):
        self.llm = llm
        self.agents = agents

    async def route_to_agent(self, request: str, top_k: int = 2) -> List[Tuple[Agent, float]]:
        """
        Route the request to the most appropriate agents.
        Returns a list of tuples containing (agent, score) for the top_k agents.
        """
        routing_prompt = f"""
        You are a routing system that determines which agents should handle a given request.
        Available agents:
        {json.dumps([agent.name for agent in self.agents], indent=2)}

        Request: {request}

        For each agent, provide a relevance score from 0 to 1 indicating how well-suited that agent is to handle this request.
        Return the scores in JSON format like this:
        {{
            "agent_name": score
        }}
        """

        # Get routing decision from LLM
        routing_decision = await self.llm.generate_str(message=routing_prompt)
        
        try:
            # Parse the routing decision
            scores = json.loads(routing_decision)
            
            # Create list of (agent, score) tuples
            agent_scores = []
            for agent in self.agents:
                if agent.name in scores:
                    agent_scores.append((agent, scores[agent.name]))
            
            # Sort by score and return top_k
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            return agent_scores[:top_k]
            
        except json.JSONDecodeError:
            # If parsing fails, return the first top_k agents
            return [(agent, 1.0) for agent in self.agents[:top_k]] 