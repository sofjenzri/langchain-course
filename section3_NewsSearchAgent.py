from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_tavily import TavilySearch




def main():
    # print("Hello from langchain-course!")

    tools = [TavilySearch()]
    react_prompt = hub.pull("hwchase17/react")
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    chain = agent_executor

    result = chain.invoke(
        input={
            "input":"rechercher les dernières actualités au sujet de la société Liot située dans le nord de la France avec des liens vers les sources pour pouvoir consulter le détail de ces actualités"
        }
    )

    print(result["output"])

    

if __name__ == "__main__":
    main()
    # print("hello")

