import webbrowser
import anthropic
import requests
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


# Tool Function: open_browser
def open_browser(url, use_chrome=False, new_window=False, new_tab=False):
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

        print(f"Successfully opened {url} in the browser.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Tool Function: brave_search
def brave_search(query, offset=0):
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return {"error": "API key not found. Please set the 'BRAVE_API_KEY' environment variable."}

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
     # Define the query parameters
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


# Tool Definitions

# Tool Definition for webbrowser_tool
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

# Tool Definition for brave_search_tool
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


# Updated Tool Definition for seleniumbase_loader
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


# Initialize Anthropic Client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Main Terminal Chat Interface with Tool Result Feedback
def chat_interface():
    print("Welcome to the Terminal Chat Interface! (Type 'exit' to quit)")

    # Base system prompt
    base_system_prompt = (
        "You are an AI assistant that can use tools like `open_browser` and `brave_search` to perform tasks. "
        "Whenever you use a tool, always respond with a confirmation message even if the tool itself does not return any result. "
        "For example, after opening a browser, say something like 'I have opened the requested URL.' "
        "After performing a search, summarize the results or state if nothing was found. "
        "This will ensure there is always a response for the user to see, and the conversation follows the correct user-assistant format."
    )

    # Initialize conversation history
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the chat. Goodbye!")
            break

        print(f"\n{'='*50}\nUser Message: {user_input}\n{'='*50}")

        # Add user input to conversation history
        conversation_history.append(f"User: {user_input}")

        # Create system prompt with conversation history
        system_prompt = (
            f"{base_system_prompt}\n\n"
            "Conversation history:\n"
            f"{' '.join(conversation_history[-5:])}"  # Include last 5 exchanges
        )

        try:
            # Send the user input to Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                tools=[webbrowser_tool, brave_search_tool,load_documents_tool,save_index_tool,load_index_tool,query_index_tool,seleniumbase_loader_tool]
            )

            print(f"\nInitial Response:")
            print(f"Stop Reason: {response.stop_reason}")
            print(f"Content: {response.content}")

            if response.stop_reason == "tool_use":
                tool_use = next(block for block in response.content if block.type == "tool_use")
                tool_name = tool_use.name
                tool_input = tool_use.input

                print(f"\nTool Used: {tool_name}")
                print(f"Tool Input: {tool_input}")

                tool_result = handle_tool_use(tool_name, tool_input)
                print(f"Tool Result: {tool_result}")

                

                # Send the tool result back to Claude
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result
                            }
                        ]}
                    ],
                    max_tokens=1024,
                    tools=[webbrowser_tool, brave_search_tool,load_documents_tool,save_index_tool,load_index_tool,query_index_tool,seleniumbase_loader_tool]
                )

            # Extract the final text response
            final_response = "\n".join([block.text for block in response.content if hasattr(block, "text")])
            print(f"\nFinal Response: {final_response}")

            # Add AI response to conversation history
            conversation_history.append(f"Assistant: {final_response}")

        except anthropic.BadRequestError as e:
            print(f"An error occurred: {e}")

# Helper function to handle tool use
def handle_tool_use(tool_name, tool_input):
    if tool_name == "open_browser":
        return open_browser(
            url=tool_input["url"],
            use_chrome=tool_input.get("use_chrome", False),
            new_window=tool_input.get("new_window", False),
            new_tab=tool_input.get("new_tab", False)
        )
    elif tool_name == "brave_search":
        # Call the brave_search function
        search_results = brave_search(
            query=tool_input["query"],
            offset=tool_input.get("offset", 0)
        )

        # If search results are a formatted string, return as-is
        if isinstance(search_results, str):
            return search_results

        # If search results are in JSON format, format the output
        if isinstance(search_results, dict) and 'web' in search_results:
            web_results = search_results['web'].get("results", [])
            if not web_results:
                return "No web results found."

            # Format the web results into a readable output
            formatted_results = "\n\n".join(
                [f"Title: {result.get('title')}\nDescription: {result.get('description', 'No description available.')}\nURL: {result['url']}"
                 for result in web_results]
            )

            return formatted_results
        else:
            return "No relevant web results found."

    elif tool_name == "summarize_web_page":
        # Handle summarize_web_page tool
        return summarize_web_page(
            url=tool_input["url"]
        )
    elif tool_name == "seleniumbase_loader":
        # Run SeleniumBase in the correct context
        return run_selenium_text_extractor(
            url=tool_input["url"],
            output_file=tool_input["output_file"],
            selector=tool_input.get("selector", None)  # Use the provided selector if available, otherwise default to None
        )
    elif tool_name == "load_documents":
        # Handle loading documents from a directory
        return load_documents(
            directory_path=tool_input["directory_path"]
        )

    elif tool_name == "save_index":
        # Handle saving the current index to disk
        return save_index(
            storage_path=tool_input["storage_path"]
        )

    elif tool_name == "load_index":
        # Handle loading an index from storage
        return load_index(
            storage_path=tool_input["storage_path"]
        )

    elif tool_name == "query_index":
        # Handle querying the loaded index
        return query_index(
            query_text=tool_input["query_text"]
        )

    else:
        return f"Error: Unknown tool '{tool_name}'"

# Start the Chat Interface
if __name__ == "__main__":
    #chat_interface()
    streamlit_chat_interface()