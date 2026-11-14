from src.agents.agent import build_search_agent, build_Reader_agent, writer_prompt, critic_promt

def run_search_pipeline( topic: str) -> dict:

    state = {} # this is dictionary to store the value, and it will be clear when agent completed

    print(f"Starting search pipeline for topic: {topic}")
    search_agent = build_search_agent()
    search_results = search_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Search the web for information about {topic} and provide relevant URLs and findings."
            }
        ]
    })

    state["search_results"] = search_results["messages"][-1].content

    print(f"Search results obtained: {state['search_results']}")



    # Start Reader Agent
    reader_agent = build_Reader_agent()

    reader_results = reader_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Read and summarize the relevant webpages about {topic}. Use the URLs found in the search results:\n{state['search_results']}"
            }
        ]
    })

    state["reader_results"] = reader_results["messages"][-1].content

    print(f"Reader results obtained: {state['reader_results']}")

    #writer_chain
    #=====================Writer chain=========================
    writer_input = {
        "topic": topic,
        "research": state["search_results"][:10000] + "\n\n" + state["reader_results"][:20000]
    }

    state["final_article"] = writer_prompt.invoke(writer_input)

    print(f"Final article generated: {state['final_article']}")

    #critic_chain
    #=====================Critic chain=========================    
    #########################################################
    #
    
    state["feedback"] = critic_promt.invoke({"report":state["final_article"] })
    
    print(f"Critic article generated: {state['feedback']}")

    return state