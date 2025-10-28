from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from fromattable_prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse


def main():
    # print("Hello from langchain-course!")

    tools = [TavilySearch()]
    react_prompt = hub.pull("hwchase17/react")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    structured_llm = llm.with_structured_output(AgentResponse)

    react_prompt_with_format_instructions = PromptTemplate(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["input", "agent_scratchpad", "tool_names"],
    ).partial(format_instructions="")

    agent = create_react_agent(
        llm=llm, tools=tools, prompt=react_prompt
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    extract_output_runnable = RunnableLambda(lambda x: x["output"])

    chain = agent_executor | extract_output_runnable | structured_llm

    result = chain.invoke(
        input={
            "input": "rechercher les dernières actualités au sujet de la société Liot située dans le nord de la France avec des liens vers les sources pour pouvoir consulter le détail de ces actualités"
        }
    )

    print(result)


if __name__ == "__main__":
    main()
    # print("hello")
