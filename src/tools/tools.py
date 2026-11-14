from langchain.tools import tool
from dotenv import load_dotenv
import os
from langchain_community.tools import TavilySearchResults
from rich import print
import requests
from bs4 import BeautifulSoup
from readability import Document
import trafilatura 
import re

load_dotenv()

tavili = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for information based on a query."""
    print(f"Searching the web for: {query}")
    response = tavili.invoke(query)

    out = []
    # print(len(response))
    for item in response:
        # print(item)
        out.append(
            "Title: " + item.get("title", "") +
            "\nURL: " + item.get("url", "") +
            "\nContent: " + item.get("content", "") +
            "\nScore: " + str(item.get("score", 0)) +
            "\n"
        )

    return "\n-----\n".join(out)

@tool
def scrape_url(url: str) -> str:
    """
    Scrape webpage content using BeautifulSoup.
    """
    print(f"Scraping URL: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary HTML elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Extract webpage title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Extract webpage text
        content = soup.get_text(separator="\n", strip=True)
        content = content[:12000]
        return (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )
    except requests.exceptions.RequestException as e:
            return f"Error scraping URL: {str(e)}"