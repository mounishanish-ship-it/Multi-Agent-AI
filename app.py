import streamlit as st
import time
#from src.pipeline.pipeline import build_search_agent, build_Reader_agent, writer_prompt,critic_promt
from src.pipeline.pipeline import run_search_pipeline 

# Page configuration
st.set_page_config(
    page_title="Multi-Agent AI Research",
    page_icon="🤖",
    layout="wide"
)

# Application title
st.title("🤖 Multi-Agent AI Research System")

st.markdown("""
This application uses multiple AI components to research a topic,
extract webpage content, generate an article, and review the output.
""")

st.divider()

# User input
st.subheader("Enter Your Research Topic")

topic = st.text_input(
    "Research Topic",
    placeholder="Example: Agentic AI, Generative AI, Machine Learning..."
)

# Run pipeline button
if st.button("🚀 Run Multi-Agent Pipeline", type="primary"):

    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            with st.spinner("Multi-Agent AI pipeline is running... Please wait!"):

                result = run_search_pipeline(topic)

            st.success("Research pipeline completed successfully!")

            # Display results
            st.divider()
            st.header("📊 Pipeline Results")

            # Search Agent
            with st.expander("🔍 Search Agent Results", expanded=True):
                st.markdown(result.get("search_results", "No search results found."))

            # Reader Agent
            with st.expander("📖 Reader Agent Results"):
                st.markdown(result.get("reader_results", "No reader results found."))

            # Writer Chain
            st.subheader("✍️ Generated Article")
            st.markdown(result.get("final_article", "No article generated."))

            # Critic Chain
            st.divider()
            st.subheader("🧐 Critic Feedback")
            st.markdown(result.get("feedback", "No feedback generated."))

            # Download article
            article = result.get("final_article", "")

            st.download_button(
                label="⬇️ Download Generated Article",
                data=article,
                file_name="research_article.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Footer
st.divider()
st.caption("Powered by LangChain | OpenAI | Tavily | BeautifulSoup | Streamlit")