
# README for Demo Project: Claude 3.5 Agent with Tool Integration

## Project Title
**Claude 3.5 Agent with Multi-Tool Integration**

## Project Description
This repository showcases a project involving a single Claude 3.5 agent integrated with multiple tools to perform a series of tasks such as web-based research, content extraction, and local document indexing. The agent interacts with the user in a prompt-driven manner, leveraging the following tools:

1. **Brave Search Tool** (`brave_search_tool`) for querying the web.
2. **Browser Opening Tool** (`open_browser`) to open URLs.
3. **Selenium-Based Web Scraper** (`seleniumbase_loader_tool`) for text extraction from specified CSS selectors.
4. **LlamaIndex Document Loader and Query Tool** (`load_documents_tool`, `save_index_tool`, `load_index_tool`, `query_index_tool`) for building and querying a local vector-based index.

### Project Goal
The project is designed to demonstrate how a single AI agent can utilize a series of specialized tools in response to user prompts to perform complex workflows like retrieving information, scraping content, and answering specific questions based on local indices. This approach simplifies automating multiple research and information-processing tasks through a single conversational interface.

## Table of Contents
1. [Project Description](#project-description)
2. [How to Install and Run the Project](#how-to-install-and-run-the-project)
3. [How to Use the Project](#how-to-use-the-project)
4. [Workflow Overview](#workflow-overview)
5. [Example Use Case](#example-use-case)

## How to Install and Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/claude-agent-tools-demo.git
   cd claude-agent-tools-demo
   ```

2. **Install the required dependencies**:
   Ensure you have Python 3.8 or higher installed.

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**: 
   ```bash
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   export BRAVE_API_KEY=your_brave_api_key
   export OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the main chat interface**: 
   Start the terminal chat interface to interact with the Claude 3.5 agent:

   ```bash
   python main.py
   ```

## How to Use the Project
- Once the terminal interface is up and running, start by typing a research query, such as **“Look up OpenAI structured outputs”**.
- The Claude 3.5 agent will:
  - Use `brave_search_tool` to find relevant web content.
  - Open a specific link in the browser using `open_browser` if requested.
  - Use `seleniumbase_loader_tool` to extract content from a specific part of the page.
  - Load the extracted text into LlamaIndex and save it for building an index.
  - Load the index and enable you to query the document for specific information.

### Workflow Overview
1. **Search for Information**:
   - Use `brave_search_tool` to look up relevant content on the web based on the user’s query.

2. **Open the Link in the Browser**:
   - The agent opens the specified link in a browser window using `open_browser`.

3. **Extract Text Using SeleniumBase**:
   - The agent uses `seleniumbase_loader_tool` to extract text from a specific CSS selector and save it to a file.

4. **Load and Save the Documents**:
   - The agent loads the saved text file into a local index using LlamaIndex for subsequent queries.

5. **Save and Load the Index**:
   - The agent saves and loads the index for future queries.

6. **Query the Index**:
   - Finally, the agent queries the index to extract answers to user queries from the stored document.

### Example Use Case
**Research Task**: Extract and analyze content about structured outputs from the OpenAI Cookbook.

1. Use `brave_search_tool` to find relevant links.
2. Open a specific link using `open_browser`.
3. Extract the main content using `seleniumbase_loader_tool`.
4. Load the document into LlamaIndex.
5. Save and load the index for querying.
6. Query the index for specific information, like benefits of structured outputs.

## Final Response for Query
I have successfully queried the index for the "Environment set up" information. Based on the response, I can provide you with the code for the environment setup. Here's the relevant code snippet:

```python
from openai import OpenAI
from IPython.display import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"
```

This code snippet sets up the environment for the multi-agent system. Here's a breakdown of what's being imported and set up:

1. `OpenAI` client is imported from the `openai` library.
2. `Image` is imported from `IPython.display` for displaying images.
3. `json` is imported for JSON data handling.
4. `pandas` is imported as `pd` for data manipulation and analysis.
5. `matplotlib.pyplot` is imported as `plt` for creating visualizations.
6. `StringIO` is imported from `io` for string I/O operations.
7. `numpy` is imported as `np` for numerical operations.

After the imports, an `OpenAI` client is instantiated, and the `MODEL` variable is set to "gpt-4o-2024-08-06", which appears to be the specific GPT-4 model version being used for this multi-agent system.
