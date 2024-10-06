import webbrowser
import requests
import json
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
import os
import subprocess
from seleniumbase import BaseCase
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chainlit as cl
import anthropic

# Set up the Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Global variables for query engine and index
query_engine = None
index = None

# Step 1: Set up API key and models globally (to be run once)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Use environment variable for API key
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set your ANTHROPIC_API_KEY as an environment variable.")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1,max_tokens=8024)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=100
)
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# Define the `webbrowser_tool` and `brave_search_tool` for Anthropic
webbrowser_tool = {
    "name": "open_browser",
    "description": "Opens a URL in a specified browser.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to be opened."},
            "use_chrome": {"type": "boolean", "description": "If True, uses Google Chrome.", "default": False},
            "new_window": {"type": "boolean", "description": "Opens in a new window.", "default": False},
            "new_tab": {"type": "boolean", "description": "Opens in a new tab.", "default": False}
        },
        "required": ["url"]
    }
}

brave_search_tool = {
    "name": "brave_search",
    "description": "Retrieves search results for a given query using Brave Search API.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "offset": {"type": "integer", "description": "Results to skip.", "default": 0}
        },
        "required": ["query"]
    }
}

# Tool Definition for seleniumbase_loader
seleniumbase_loader_tool = {
    "name": "seleniumbase_loader",
    "description": (
        "Loads a URL, extracts text content using SeleniumBase, and saves it to a text file. "
        "Useful for scraping the full text content of a webpage and storing it locally for analysis or reference. "
        "Optionally, a CSS selector can be provided to specify the section of content to extract. If not provided, it defaults to predefined main content selectors."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to extract text content from. Must be a valid URL starting with http:// or https://.",
            },
            "output_file": {
                "type": "string",
                "description": (
                    "The path to the output text file where the extracted content will be saved. "
                    "Ensure the file path is accessible and valid."
                )
            },
            "selector": {
                "type": "string",
                "description": (
                    "Optional. A specific CSS selector to extract content from. "
                    "If not provided, a set of default main content selectors will be used."
                ),
                "default": None
            }
        },
        "required": ["url", "output_file"]
    }
}

# Tool Definition for load_documents
load_documents_tool = {
    "name": "load_documents",
    "description": "Loads documents from the specified directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "The path to the directory containing the text documents to load.",
            }
        },
        "required": ["directory_path"]
    }
}

# Tool Definition for save_index
save_index_tool = {
    "name": "save_index",
    "description": "Saves the current index to a specified directory path.",
    "input_schema": {
        "type": "object",
        "properties": {
            "storage_path": {
                "type": "string",
                "description": "The directory path to save the current index.",
            }
        },
        "required": ["storage_path"]
    }
}

# Tool Definition for load_index
load_index_tool = {
    "name": "load_index",
    "description": "Loads an index from a specified storage directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "storage_path": {
                "type": "string",
                "description": "The path to the directory where the index is stored.",
            }
        },
        "required": ["storage_path"]
    }
}

# Tool Definition for query_index
query_index_tool = {
    "name": "query_index",
    "description": "Queries the loaded index and returns a response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_text": {
                "type": "string",
                "description": "The query text to search in the index.",
            }
        },
        "required": ["query_text"]
    }
}

# Tool Function: open_browser
def open_browser(url, use_chrome=False, new_window=False, new_tab=False):
    """Open a URL in the specified browser."""
    try:
        if use_chrome:
            browser = webbrowser.Chrome('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        else:
            browser = webbrowser.get()
        
        if new_window:
            browser.open_new(url)
        elif new_tab:
            browser.open_new_tab(url)
        else:
            browser.open(url)

        return f"Successfully opened {url} in the browser."
    except Exception as e:
        return f"An error occurred: {e}"

# Tool Function: brave_search
def brave_search(query, offset=0):
    """Retrieve search results using Brave Search API."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return {"error": "API key not found. Please set the 'BRAVE_API_KEY' environment variable."}

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "offset": offset,
        "result_filter": "web"  # Use result_filter to include only web results
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to retrieve data, status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# SeleniumBase Test Class for Text Extraction with Optional Selector Support
class SeleniumTextExtractor(BaseCase):
    def seleniumbase_loader(self, url, output_file, selector=None):
        """
        Loads a URL, extracts the main content using either a provided selector or a refined set of predefined selectors, and saves it to a text file.

        Parameters:
        - url (str): The URL to extract main content from.
        - output_file (str): The path to the output text file.
        - selector (str, optional): The CSS selector to directly extract content. If not provided, the function will fall back to predefined content selectors.

        Returns:
        - str: Success message indicating the text was saved to the specified file.
        """
        # Step 1: Open the specified URL
        self.open(url)

        # Step 2: Remove unwanted elements (styles, scripts, and metadata)
        self.remove_elements("style, script, header, footer, nav, aside, meta, link")  # Remove unwanted tags

        # Step 3: If a custom selector is provided, extract content using that
        if selector:
            try:
                page_text = self.get_text(selector)
            except Exception:
                page_text = f"Failed to extract content using the provided selector: '{selector}'"
        else:
            # Define a set of CSS selectors to capture the main content
            content_selectors = [
                "article",        # Commonly used for articles or main text
                "main",           # Main content wrapper in most modern websites
                "section",        # Section tags often wrap main content
                "div.content",    # Generic class-based selectors
                "div.article-body",
                "div.post-content",
                "div#content",    # IDs for main content sections
                "div#main-content",
            ]

            # Step 4: Attempt to extract text content from the defined selectors
            page_text = ""
            for default_selector in content_selectors:
                try:
                    page_text = self.get_text(default_selector)
                    if page_text.strip():  # If we successfully capture some text, stop
                        break
                except Exception:
                    continue

            # Fallback: If no main content is found, get the whole HTML text without styles and scripts
            if not page_text.strip():
                page_text = self.get_text("body")

        # Step 5: Save the cleaned, extracted text to the specified output file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(page_text)

        return f"Successfully saved main content from '{url}' to '{output_file}'."


# Define a Test Runner Function Using SeleniumBase and pytest with Optional Selector Support
def run_selenium_text_extractor(url, output_file, selector=None):
    """
    Runs the SeleniumBase loader function using a temporary script and pytest with optional selector support.

    Parameters:
    - url (str): The URL to extract main content from.
    - output_file (str): The path to the output text file.
    - selector (str, optional): The CSS selector to directly extract content. If not provided, uses predefined set of selectors.

    Returns:
    - str: Success message from SeleniumBase indicating the text was saved.
    """
    selector_code = f'self.get_text("{selector}")' if selector else """
        content_selectors = [
            "article", "main", "section", "div.content", 
            "div.article-body", "div.post-content", "div#content", "div#main-content"
        ]
        page_text = ""
        for sel in content_selectors:
            try:
                page_text = self.get_text(sel)
                if page_text.strip():
                    break
            except Exception:
                continue
        if not page_text.strip():
            page_text = self.get_text("body")
    """

    temp_script = f"""
from seleniumbase import BaseCase

class TempTextExtractor(BaseCase):
    def test_extract_main_text(self):
        self.open("{url}")
        self.remove_elements("style, script, header, footer, nav, aside, meta, link")
        page_text = {selector_code}
        with open("{output_file}", "w", encoding="utf-8") as file:
            file.write(page_text)
        print("Successfully saved content from '{url}' to '{output_file}'.")

if __name__ == "__main__":
    from pytest import main
    main(["-v", __file__])
"""
    temp_script_path = "temp_selenium_test.py"

    with open(temp_script_path, "w") as file:
        file.write(temp_script)

    try:
        subprocess.run(["pytest", temp_script_path], check=True)
        return f"Successfully saved content from '{url}' to '{output_file}'."
    except subprocess.CalledProcessError as e:
        return f"Failed to extract text from '{url}'. Error: {e}"
    finally:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)


# Tool Function: Load Documents from a Directory
def load_documents(directory_path):
    """
    Loads documents from the specified directory.

    Parameters:
    - directory_path (str): Path to the directory containing text files.

    Returns:
    - str: Success or error message.
    """
    global index

    # Check if the directory is valid
    if not os.path.isdir(directory_path):
        return f"Error: The directory '{directory_path}' is not valid."

    # Load the documents from the specified directory
    documents = SimpleDirectoryReader(directory_path).load_data()
    if not documents:
        return f"Error: No documents found in directory '{directory_path}'."
    
    # Create a new index with the loaded documents
    index = VectorStoreIndex.from_documents(documents)
    return f"Successfully loaded {len(documents)} documents from '{directory_path}'."


# Tool Function: Save Index to Disk
def save_index(storage_path):
    """
    Saves the current index to disk.

    Parameters:
    - storage_path (str): Path to save the index.

    Returns:
    - str: Success or error message.
    """
    global index
    if not index:
        return "Error: No index is currently loaded. Please create or load an index first."

    # Create storage context and save index
    index.set_index_id("vector_index")
    index.storage_context.persist(storage_path)
    return f"Index successfully saved to '{storage_path}'."


# Tool Function: Load Index from Storage
def load_index(storage_path):
    """
    Loads an index from the specified storage directory.

    Parameters:
    - storage_path (str): Path to the directory where the index is stored.

    Returns:
    - str: Success or error message.
    """
    global index, query_engine

    # Check if the storage path exists
    if not os.path.isdir(storage_path):
        return f"Error: The storage path '{storage_path}' does not exist."

    # Load the index from storage
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context, index_id="vector_index")
        query_engine = index.as_query_engine(similarity_top_k=3)
        return f"Index successfully loaded from '{storage_path}'."
    except Exception as e:
        return f"Error loading index: {e}"



# Tool Function: Query the Loaded Index
def query_index(query_text):
    """
    Queries the loaded index and returns a response.

    Parameters:
    - query_text (str): The text query to search in the index.

    Returns:
    - str: The response from the query engine.
    """
    global query_engine

    # Ensure query_text is a string
    if isinstance(query_text, dict) and 'query_text' in query_text:
        query_text = query_text['query_text']

    if not query_engine:
        return "Error: Query engine is not set up. Please load an index first."

    # Pass the query_text directly as a string
    try:
        response = query_engine.query(query_text)
        print(response)
        return f"Query: {query_text}\nResponse: {response.source_nodes}"
    except Exception as e:
        return f"An error occurred: {e}"



# Define tool options to pass to Claude
tools = [webbrowser_tool, brave_search_tool,seleniumbase_loader_tool,load_documents_tool,save_index_tool,load_index_tool,query_index_tool]

@cl.on_chat_start
async def start_chat():
    """Initialize conversation history at the start of the chat."""
    cl.user_session.set("message_history", [])


# Step function to handle tool calls
@cl.step(type="tool")
async def call_tool(tool_call, message_history):
    """Handle tool execution based on tool_call data."""
    tool_name = tool_call.name
    arguments = tool_call.input  # Directly access the input field

    print(f"\n[DEBUG] Tool Call - Name: {tool_name}, Arguments: {arguments}")

    current_step = cl.context.current_step
    current_step.name = tool_name
    current_step.input = arguments

    if tool_name == "open_browser":
        tool_result = open_browser(
            url=arguments["url"],
            use_chrome=arguments.get("use_chrome", False),
            new_window=arguments.get("new_window", False),
            new_tab=arguments.get("new_tab", False)
        )
    elif tool_name == "brave_search":
        search_results = brave_search(
            query=arguments["query"],
            offset=arguments.get("offset", 0)
        )
        if isinstance(search_results, dict) and 'web' in search_results:
            web_results = search_results['web'].get("results", [])
            if not web_results:
                tool_result = "No web results found."
            else:
                tool_result = "\n\n".join(
                    [f"Title: {result.get('title')}\nDescription: {result.get('description', 'No description available.')}\nURL: {result['url']}"
                     for result in web_results]
                )
        else:
            tool_result = search_results
    elif tool_name == "seleniumbase_loader":
        tool_result = run_selenium_text_extractor(
            url=arguments["url"],
            output_file=arguments["output_file"],
            selector=arguments.get("selector",None)
        )
    elif tool_name == "load_documents":
        tool_result = load_documents(arguments["directory_path"])
    elif tool_name == "save_index":
        tool_result = save_index(arguments["storage_path"])
    elif tool_name == "load_index":
        tool_result = load_index(arguments["storage_path"])
    elif tool_name == "query_index":
        tool_result = query_index(arguments["query_text"])
    else:
        tool_result = f"Error: Unknown tool '{tool_name}'"

    current_step.output = tool_result
    current_step.language = "text"

    
    return tool_result

async def call_claude(query: str):
    """Send the user's query to Claude and handle the response."""
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": query})

    print(f"\n[DEBUG] Message History Before Sending to Claude:\n{json.dumps(message_history, indent=2)}")

    system_prompt = """
    You are a helpful assistant. When providing information based on search results, 
    always include all the relevant links from the Brave search results in your response. 
    List these links at the end of your response, even if you haven't directly referenced 
    all of them in your main explanation.

    You can also use the seleniumbase_loader tool to extract content from web pages. 
    This is particularly useful when you need to analyze or reference the full content 
    of a specific webpage. When using this tool, make sure to provide a meaningful 
    output file name and consider using a specific CSS selector if you want to target 
    particular content on the page.

    You also have access to the following LlamaIndex tools:
    - load_documents: Load documents from a specified directory.
    - save_index: Save the current index to a specified directory.
    - load_index: Load an index from a specified storage directory.
    - query_index: Query the loaded index with a given text.

    Use these tools when you need to work with document collections or perform 
    semantic searches on indexed content.
    """

    settings = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": message_history,
        "max_tokens": 4096,
        "tools": tools,
        "system": system_prompt,
    }

    # Send the initial query to Claude
    response = await client.messages.create(**settings)

    print(f"\n[DEBUG] Raw Response from Claude:\n{response}")

    # Check if Claude wants to use a tool
    if response.stop_reason == "tool_use":
        tool_call = next(block for block in response.content if block.type == "tool_use")

        print(f"\n[DEBUG] Tool Use Block Details:\n{json.dumps(tool_call.model_dump(), indent=2)}")

        # Execute the tool based on the provided input
        tool_result = await call_tool(tool_call, message_history)

        print(f"\n[DEBUG] Tool Result:\n{tool_result}")

        # Append Claude's response (including tool use) to the message history
        message_history.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content]
        })

        # Append the tool result as a user message
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_result
                }
            ]
        })

        print(f"\n[DEBUG] Message History Before Final Request:\n{json.dumps(message_history, indent=2)}")

        # Send back the updated message history to Claude for the final response
        final_response = await client.messages.create(
            model=settings["model"],
            messages=message_history,
            max_tokens=settings["max_tokens"],
            tools=settings["tools"],
            system=system_prompt
        )

        # Extract and format the final response text
        final_text_response = "\n".join([block.text for block in final_response.content if hasattr(block, "text")])

        # Update the message history and session
        message_history.append({"role": "assistant", "content": final_text_response})
        cl.user_session.set("message_history", message_history)

        print(f"\n[DEBUG] Final Text Response:\n{final_text_response}")

        return final_text_response

    # If no tool use is required, return the initial response text
    initial_response_text = "\n".join([block.text for block in response.content if hasattr(block, "text")])
    message_history.append({"role": "assistant", "content": initial_response_text})
    cl.user_session.set("message_history", message_history)

    print(f"\n[DEBUG] Initial Response Text:\n{initial_response_text}")

    return initial_response_text





# Chat function to handle incoming user messages
@cl.on_message
async def chat(message: cl.Message):
    """Handle incoming user messages."""
    # Get the response from Claude
    final_content = await call_claude(message.content)

    # Send the final message to the Chainlit UI
    await cl.Message(content=final_content, author="Claude").send()