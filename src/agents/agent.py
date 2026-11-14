from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o",temperature=0, max_tokens=4000)

# creating search agent
def build_search_agent():
   
    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=""
       
    )
    return agent

#creating second agent
def build_Reader_agent():
   
    agent = create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=""
     
    )
    return agent

#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant and an expert content writer."),

    ("user", """
    Write a comprehensive article on {topic}
    based on the following research data:

    Research Data:
    {research}

   
    Instructions:
    - Use clear and simple language.
    - Include relevant facts from the research.
    - Maintain a professional writing style.
    - Do not invent information.
    - Organize the report with proper headings.
    """)
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 
critic_promt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant and an expert content critic."),

    ("user", """
    Critique the following article based on the research data provided:

    Report:
    {report}


    Instructions:
    - Check for accuracy of information.
    - Ensure all claims are supported by the research.
    - Provide constructive feedback for improvement.
    - Highlight any discrepancies or missing information.
    """)
])
critic_chain = critic_promt | llm | StrOutputParser()