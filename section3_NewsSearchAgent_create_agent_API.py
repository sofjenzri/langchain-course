from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse


def main():
    # print("Hello from langchain-course!")

    # Initialize tools and model
    tools = [TavilySearch()]
    model = ChatOpenAI(temperature=0, model="gpt-4o")

    # Create agent with structured output
    agent = create_agent(
        model=model,
        tools=tools,
        response_format=ProviderStrategy(AgentResponse)
    )

    # Invoke the agent
    result = agent.invoke(
        {
            "messages": [{
                "role": "user", 
                "content": "rechercher les dernières actualités au sujet de la société Liot située dans le nord de la France avec des liens vers les sources pour pouvoir consulter le détail de ces actualités"
            }]
        }
    )

    print(result)


if __name__ == "__main__":
    main()
    # print("hello")
