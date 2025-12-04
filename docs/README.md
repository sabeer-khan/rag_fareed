Retrieval Augmented Generation (RAG) is a design pattern that augments the capabilities of a chat completion model like ChatGPT by adding an information retrieval step, incorporating your proprietary enterprise content for answer formulation. For an enterprise solution, it's possible to fully constrain generative AI to your enterprise content.

The decision about which information retrieval system to use is critical because it determines the inputs to the LLM. The information retrieval system should provide:

    Indexing strategies that load and refresh at scale, for all of your content, at the frequency you require.

    Query capabilities and relevance tuning. The system should return relevant results, in the short-form formats necessary for meeting the token length requirements of large language model (LLM) inputs. Query turnaround should be as fast as possible.

    Security, global reach, and reliability for both data and operations.

    Integration with embedding models for indexing, and chat models or language understanding models for retrieval.

What is Retrieval-Augmented Generation?
Retrieval-Augmented Generation (RAG) optimizes the output of a large language model so that it references an authoritative knowledge base outside its training data sources before generating an answer. Large language models (LLMs) are trained on massive datasets and use billions of parameters to generate original outputs for tasks such as answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base without requiring the model to be retrained. It is a cost-effective approach to improving LLM results so that they remain relevant, accurate, and useful across different contexts.

Why is Retrieval-Augmented Generation important?
Learning Learning Modules (LLMs) are an important technology for artificial intelligence (AI) , supporting intelligent chatbots and other natural language processing (NLP) applications . The goal is to develop bots that can answer user questions in various contexts by referencing authoritative knowledge sources. Unfortunately, the nature of LLM technology leads to unpredictability in LLM responses. Furthermore, LLM training data is static, introducing a cutoff date for the available knowledge.

Known challenges faced by LLMs include:

Present false information when no response is received.
Presenting outdated or general information when the user expects a specific, up-to-date answer.
Create an answer from non-authoritative sources.
Inaccurate answers arise due to confusing terminology, where different training sources use the same terminology to talk about different things.
You can think of the large language model as an overly enthusiastic new employee who refuses to stay up-to-date on current events but always answers every question with absolute confidence. Unfortunately, such an attitude can negatively impact user trust, and you don't want your chatbots to mimic that!

RAG is one approach to solving some of these challenges. It redirects the LLM to retrieve relevant information from authoritative, predefined knowledge sources. Organizations have better control over the generated text output, and users gain insight into how the LLM generates the response.

What are the advantages of retrieval augmented generation?
RAG technology offers several advantages for a company's generative AI efforts .

Cost-effective implementation
Chatbot development typically begins with a base model . Base models (FMs) are API-accessible LLMs trained on a broad range of generalized and unlabeled data. Retraining FMs for organization- or domain-specific information is computationally and financially costly. RAG (Reinforced Agility Grouping) is a more cost-effective approach to introducing new data into the LLM. It makes generative artificial intelligence (generative AI) technology more widely accessible and usable.

Current information
Even if the original training data sources for an LLM are suitable for your needs, maintaining relevance can be challenging. RAG enables developers to provide the latest research, statistics, or news about the generative models. You can use RAG to connect the LLM directly to live feeds on social media, news sites, or other frequently updated information sources. The LLM can then provide users with the most up-to-date information.

Improved user trust
RAG enables the LLM to present accurate information with source citations. The output can include citations or references to sources. Users can also look up source documents themselves if they need further explanation or details. This can increase confidence in your generative AI solution.

More control by developers
With RAG, developers can test and improve their chat applications more efficiently. They can control and modify the LLM's information sources to adapt to changing requirements or cross-functional use. Developers can also restrict access to sensitive information to different authorization levels and ensure the LLM generates appropriate responses. Furthermore, they can troubleshoot and correct errors if the LLM refers to incorrect information sources for certain questions. Organizations can more securely implement generative AI technology for a wider range of applications.

How does retrieval-augmented generation work?
Without RAG, the LLM takes user input and generates a response based on information it has been trained on—or on what it already knows. RAG introduces an information retrieval component that uses the user input to first retrieve information from a new data source. The user request and the relevant information are both passed to the LLM. The LLM uses this new knowledge and its training data to find better answers. The following sections provide an overview of the process.

Create external data
New data outside the original LLM training dataset is referred to as external data . It can originate from multiple data sources, such as APIs, databases, or document repositories. The data can be in various formats, including files, database records, or long-form text. Another AI technique, language model embedding , converts data into numerical representations and stores them in a vector database. This process creates a knowledge library that generative AI models can understand.

Retrieving relevant information
The next step is to perform a relevance search. The user query is converted into a vector representation and matched against vector databases. For example, imagine an intelligent chatbot that can answer HR questions for a company. If an employee asks, "How much annual leave do I have?", the system retrieves the annual leave documents along with the employee's records of past leave. These specific documents are returned because they are highly relevant to the employee's input. The relevance was calculated and determined using mathematical vector calculations and representations.

Extend the LLM prompt
Next, the RAG model augments user input (or prompts) by adding the relevant retrieved data to the context. This step employs prompt engineering techniques to communicate effectively with the LLM. The augmented prompt enables the large language models to generate accurate responses to user queries.

Update external data
The next question might be: What if the external data is outdated? To maintain current information for retrieval, update the documents asynchronously and refresh the document embedding representation. You can do this through automated, real-time processes or regular batch processing. This is a common challenge in data analytics—various data science approaches to change management can be used.

The following diagram shows the conceptual flow of using RAG with LLMs.


 

What is the difference between retrieval augmented generation and semantic search?
Semantic search improves RAG results for organizations that want to add extensive external knowledge sources to their LLM applications. Modern organizations store vast amounts of information—such as manuals, FAQs, research reports, customer service guides, and personnel documentation—across various systems. Retrieving context at scale is challenging and consequently impacts the quality of generative output.

Semantic search technologies can scan large databases containing diverse information and retrieve data more precisely. For example, they can answer questions like, "How much was spent on machine repairs last year?" by mapping the question to relevant documents and returning specific text instead of search results. Developers can then use this answer to provide more context to the LLM (Learning Management Module).

Conventional or keyword-based search solutions in RAG deliver limited results for knowledge-intensive tasks. Developers also have to deal with word embedding, document splitting, and other complexities when manually preparing their data. In contrast, semantic search technologies handle all the work of knowledge base preparation, eliminating the need for developers. They also generate semantically relevant passages and relevance-ordered token words to maximize the quality of the RAG payload.

How can AWS support your retrieval-augmented generation requirements?
Amazon Bedrock is a fully managed service that offers a selection of powerful base models and a wide range of features for building generative AI applications, while simplifying development and ensuring data privacy and security. With Amazon Bedrock knowledge bases, you can connect FMs to your RAG data sources with just a few clicks. Vector conversions, retrievals, and enhanced output generation are all handled automatically.

For organizations that manage their own RAG (Research Area Group), Amazon Kendra is a highly accurate enterprise search service powered by machine learning. It offers a streamlined Kendra Retrieve API that you can use, along with Amazon Kendra's highly accurate semantic ranker, as an enterprise retriever for your RAG workflows. With the Retrieve API, you can, for example:

Retrieve up to 100 semantically relevant passages, each containing up to 200 token words, sorted by relevance.
Use pre-built connectors for popular data technologies such as Amazon Simple Storage Service , SharePoint, Confluence, and other websites.
Supports a wide variety of document formats such as HTML, Word, PowerPoint, PDF, Excel and text files.

LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and more. LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.
We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use LangGraph, our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.
LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.
​
 Create an agent

Copy
# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
See the Installation instructions and Quickstart guide to get started building your own agents and applications with LangChain.


Install LangChain

Copy page

To install the LangChain package:

pip

uv

Copy
pip install -U langchain
# Requires Python 3.10+
LangChain provides integrations to hundreds of LLMs and thousands of other integrations. These live in independent provider packages. For example:

pip

uv

Copy
# Installing the OpenAI integration
pip install -U langchain-openai

# Installing the Anthropic integration
pip install -U langchain-anthropic


Agents

Copy page

Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions.
create_agent provides a production-ready agent implementation.
An LLM Agent runs tools in a loop to achieve a goal. An agent runs until a stop condition is met - i.e., when the model emits a final output or an iteration limit is reached.
action

observation

finish

input

model

tools

output

create_agent builds a graph-based agent runtime using LangGraph. A graph consists of nodes (steps) and edges (connections) that define how your agent processes information. The agent moves through this graph, executing nodes like the model node (which calls the model), the tools node (which executes tools), or middleware.
Learn more about the Graph API.
​
Core components
​
Model
The model is the reasoning engine of your agent. It can be specified in multiple ways, supporting both static and dynamic model selection.
​
Static model
Static models are configured once when creating the agent and remain unchanged throughout execution. This is the most common and straightforward approach.
To initialize a static model from a model identifier string:

Copy
from langchain.agents import create_agent

agent = create_agent(
    "gpt-5",
    tools=tools
)
Model identifier strings support automatic inference (e.g., "gpt-5" will be inferred as "openai:gpt-5"). Refer to the reference to see a full list of model identifier string mappings.
For more control over the model configuration, initialize a model instance directly using the provider package. In this example, we use ChatOpenAI. See Chat models for other available chat model classes.

Copy
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)
agent = create_agent(model, tools=tools)
Model instances give you complete control over configuration. Use them when you need to set specific parameters like temperature, max_tokens, timeouts, base_url, and other provider-specific settings. Refer to the reference to see available params and methods on your model.
​
Dynamic model
Dynamic models are selected at runtime based on the current state and context. This enables sophisticated routing logic and cost optimization.
To use a dynamic model, create middleware using the @wrap_model_call decorator that modifies the model in the request:

Copy
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]



    Models

Copy page

LLMs are powerful AI tools that can interpret and generate text like humans. They’re versatile enough to write content, translate languages, summarize, and answer questions without needing specialized training for each task.
In addition to text generation, many models support:
 Tool calling - calling external tools (like databases queries or API calls) and use results in their responses.
 Structured output - where the model’s response is constrained to follow a defined format.
 Multimodality - process and return data other than text, such as images, audio, and video.
 Reasoning - models perform multi-step reasoning to arrive at a conclusion.
Models are the reasoning engine of agents. They drive the agent’s decision-making process, determining which tools to call, how to interpret results, and when to provide a final answer.
The quality and capabilities of the model you choose directly impact your agent’s baseline reliability and performance. Different models excel at different tasks - some are better at following complex instructions, others at structured reasoning, and some support larger context windows for handling more information.
LangChain’s standard model interfaces give you access to many different provider integrations, which makes it easy to experiment with and switch between models to find the best fit for your use case.
For provider-specific integration information and capabilities, see the provider’s chat model page.
​
Basic usage
Models can be utilized in two ways:
With agents - Models can be dynamically specified when creating an agent.
Standalone - Models can be called directly (outside of the agent loop) for tasks like text generation, classification, or extraction without the need for an agent framework.
The same model interface works in both contexts, which gives you the flexibility to start simple and scale up to more complex agent-based workflows as needed.
​
Initialize a model
The easiest way to get started with a standalone model in LangChain is to use init_chat_model to initialize one from a chat model provider of your choice (examples below):
OpenAI
Anthropic
Azure
Google Gemini
AWS Bedrock
👉 Read the OpenAI chat model integration docs

Copy
pip install -U "langchain[openai]"

init_chat_model

Model Class

Copy
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-4.1")

Copy
response = model.invoke("Why do parrots talk?")
See init_chat_model for more detail, including information on how to pass model parameters.
​
Key methods
Invoke
The model takes messages as input and outputs messages after generating a complete response.
Stream
Invoke the model, but stream the output as it is generated in real-time.
Batch
Send multiple requests to a model in a batch for more efficient processing.
In addition to chat models, LangChain provides support for other adjacent technologies, such as embedding models and vector stores. See the integrations page for details.
​
Parameters
A chat model takes parameters that can be used to configure its behavior. The full set of supported parameters varies by model and provider, but standard ones include:
​
model
stringrequired
The name or identifier of the specific model you want to use with a provider. You can also specify both the model and its provider in a single argument using the ’:’ format, for example, ‘openai:o1’.
​
api_key
string
The key required for authenticating with the model’s provider. This is usually issued when you sign up for access to the model. Often accessed by setting an environment variable.
​
temperature
number
Controls the randomness of the model’s output. A higher number makes responses more creative; lower ones make them more deterministic.
​
max_tokens
number
Limits the total number of tokens in the response, effectively controlling how long the output can be.
​
timeout
number
The maximum time (in seconds) to wait for a response from the model before canceling the request.
​
max_retries
number
The maximum number of attempts the system will make to resend a request if it fails due to issues like network timeouts or rate limits.
Using init_chat_model, pass these parameters as inline **kwargs:
Initialize using model parameters

Copy
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)
Each chat model integration may have additional params used to control provider-specific functionality.
For example, ChatOpenAI has use_responses_api to dictate whether to use the OpenAI Responses or Completions API.
To find all the parameters supported by a given chat model, head to the chat model integrations page.
​
Invocation
A chat model must be invoked to generate an output. There are three primary invocation methods, each suited to different use cases.
​
Invoke
The most straightforward way to call a model is to use invoke() with a single message or a list of messages.
Single message

Copy
response = model.invoke("Why do parrots have colorful feathers?")
print(response)
A list of messages can be provided to a chat model to represent conversation history. Each message has a role that models use to indicate who sent the message in the conversation.
See the messages guide for more detail on roles, types, and content.
Dictionary format

Copy
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")
Message objects

Copy
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")
If the return type of your invocation is a string, ensure that you are using a chat model as opposed to a LLM. Legacy, text-completion LLMs return strings directly. LangChain chat models are prefixed with “Chat”, e.g., ChatOpenAI(/oss/integrations/chat/openai).
​
Stream
Most models can stream their output content while it is being generated. By displaying output progressively, streaming significantly improves user experience, particularly for longer responses.
Calling stream() returns an iterator that yields output chunks as they are produced. You can use a loop to process each chunk in real-time:

Basic text streaming

Stream tool calls, reasoning, and other content

Copy
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
As opposed to invoke(), which returns a single AIMessage after the model has finished generating its full response, stream() returns multiple AIMessageChunk objects, each containing a portion of the output text. Importantly, each chunk in a stream is designed to be gathered into a full message via summation:
Construct an AIMessage

Copy
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]
The resulting message can be treated the same as a message that was generated with invoke() – for example, it can be aggregated into a message history and passed back to the model as conversational context.
Streaming only works if all steps in the program know how to process a stream of chunks. For instance, an application that isn’t streaming-capable would be one that needs to store the entire output in memory before it can be processed.
Advanced streaming topics

​
Batch
Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can be done in parallel:
Batch

Copy
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
This section describes a chat model method batch(), which parallelizes model calls client-side.
It is distinct from batch APIs supported by inference providers, such as OpenAI or Anthropic.
By default, batch() will only return the final output for the entire batch. If you want to receive the output for each individual input as it finishes generating, you can stream results with batch_as_completed():
Yield batch responses upon completion

Copy
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)
When using batch_as_completed(), results may arrive out of order. Each includes the input index for matching to reconstruct the original order as needed.
When processing a large number of inputs using batch() or batch_as_completed(), you may want to control the maximum number of parallel calls. This can be done by setting the max_concurrency attribute in the RunnableConfig dictionary.
Batch with max concurrency

Copy
model.batch(
    list_of_inputs,
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls
    }
)
See the RunnableConfig reference for a full list of supported attributes.
For more details on batching, see the reference.
​
Tool calling
Models can request to call tools that perform tasks such as fetching data from a database, searching the web, or running code. Tools are pairings of:
A schema, including the name of the tool, a description, and/or argument definitions (often a JSON schema)
A function or coroutine to execute.
You may hear the term “function calling”. We use this interchangeably with “tool calling”.
Here’s the basic tool calling flow between a user and a model:
Tools
Model
User
Tools
Model
User
par
[Parallel Tool Calls]
par
[Tool Execution]
"What's the weather in SF and NYC?"
Analyze request & decide tools needed
get_weather("San Francisco")
get_weather("New York")
SF weather data
NYC weather data
Process results & generate response
"SF: 72°F sunny, NYC: 68°F cloudy"
To make tools that you have defined available for use by a model, you must bind them using bind_tools. In subsequent invocations, the model can choose to call any of the bound tools as needed.
Some model providers offer built-in tools that can be enabled via model or invocation parameters (e.g. ChatOpenAI, ChatAnthropic). Check the respective provider reference for details.
See the tools guide for details and other options for creating tools.
Binding user tools

Copy
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])  

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
When binding user-defined tools, the model’s response includes a request to execute a tool. When using a model separately from an agent, it is up to you to execute the requested tool and return the result back to the model for use in subsequent reasoning. When using an agent, the agent loop will handle the tool execution loop for you.
Below, we show some common ways you can use tool calling.
Tool execution loop

Forcing tool calls

Parallel tool calls

Streaming tool calls

​
Structured output
Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent processing. LangChain supports multiple schema types and methods for enforcing structured output.
Pydantic
TypedDict
JSON Schema
Pydantic models provide the richest feature set with field validation, descriptions, and nested structures.

Copy
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
Key considerations for structured output:
Method parameter: Some providers support different methods ('json_schema', 'function_calling', 'json_mode')
'json_schema' typically refers to dedicated structured output features offered by a provider
'function_calling' derives structured output by forcing a tool call following the given schema
'json_mode' is a precursor to 'json_schema' offered by some providers - it generates valid json, but the schema must be described in the prompt
Include raw: Use include_raw=True to get both the parsed output and the raw AI message
Validation: Pydantic models provide automatic validation, while TypedDict and JSON Schema require manual validation
Example: Message output alongside parsed structure

Example: Nested structures

​
Supported models
LangChain supports all major model providers, including OpenAI, Anthropic, Google, Azure, AWS Bedrock, and more. Each provider offers a variety of models with different capabilities. For a full list of supported models in LangChain, see the integrations page.
​
Advanced topics
​
Model profiles
This is a beta feature. The format of model profiles is subject to change.
Model profiles require langchain>=1.1.
LangChain chat models can expose a dictionary of supported features and capabilities through a .profile attribute:

Copy
model.profile
# {
#   "max_input_tokens": 400000,
#   "image_inputs": True,
#   "reasoning_output": True,
#   "tool_calling": True,
#   ...
# }
Refer to the full set of fields in the API reference.
Much of the model profile data is powered by the models.dev project, an open source initiative that provides model capability data. These data are augmented with additional fields for purposes of use with LangChain. These augmentations are kept aligned with the upstream project as it evolves.
Model profile data allow applications to work around model capabilities dynamically. For example:
Summarization middleware can trigger summarization based on a model’s context window size.
Structured output strategies in create_agent can be inferred automatically (e.g., by checking support for native structured output features).
Model inputs can be gated based on supported modalities and maximum input tokens.
Updating or overwriting profile data

​
Multimodal
Certain models can process and return non-textual data such as images, audio, and video. You can pass non-textual data to a model by providing content blocks.
All LangChain chat models with underlying multimodal capabilities support:
Data in the cross-provider standard format (see our messages guide)
OpenAI chat completions format
Any format that is native to that specific provider (e.g., Anthropic models accept Anthropic native format)
See the multimodal section of the messages guide for details.
Some models can return multimodal data as part of their response. If invoked to do so, the resulting AIMessage will have content blocks with multimodal types.
Multimodal output

Copy
response = model.invoke("Create a picture of a cat")
print(response.content_blocks)
# [
#     {"type": "text", "text": "Here's a picture of a cat"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]
See the integrations page for details on specific providers.
​
Reasoning
Many models are capable of performing multi-step reasoning to arrive at a conclusion. This involves breaking down complex problems into smaller, more manageable steps.
If supported by the underlying model, you can surface this reasoning process to better understand how the model arrived at its final answer.

Stream reasoning output

Complete reasoning output

Copy
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)
Depending on the model, you can sometimes specify the level of effort it should put into reasoning. Similarly, you can request that the model turn off reasoning entirely. This may take the form of categorical “tiers” of reasoning (e.g., 'low' or 'high') or integer token budgets.
For details, see the integrations page or reference for your respective chat model.
​
Local models
LangChain supports running models locally on your own hardware. This is useful for scenarios where either data privacy is critical, you want to invoke a custom model, or when you want to avoid the costs incurred when using a cloud-based model.
Ollama is one of the easiest ways to run models locally. See the full list of local integrations on the integrations page.
​
Prompt caching
Many providers offer prompt caching features to reduce latency and cost on repeat processing of the same tokens. These features can be implicit or explicit:
Implicit prompt caching: providers will automatically pass on cost savings if a request hits a cache. Examples: OpenAI and Gemini.
Explicit caching: providers allow you to manually indicate cache points for greater control or to guarantee cost savings. Examples:
ChatOpenAI (via prompt_cache_key)
Anthropic’s AnthropicPromptCachingMiddleware
Gemini.
AWS Bedrock
Prompt caching is often only engaged above a minimum input token threshold. See provider pages for details.
Cache usage will be reflected in the usage metadata of the model response.
​
Server-side tool use
Some providers support server-side tool-calling loops: models can interact with web search, code interpreters, and other tools and analyze the results in a single conversational turn.
If a model invokes a tool server-side, the content of the response message will include content representing the invocation and result of the tool. Accessing the content blocks of the response will return the server-side tool calls and results in a provider-agnostic format:
Invoke with server-side tool use

Copy
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini")

tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke("What was a positive news story from today?")
response.content_blocks
Result

Copy
[
    {
        "type": "server_tool_call",
        "name": "web_search",
        "args": {
            "query": "positive news stories today",
            "type": "search"
        },
        "id": "ws_abc123"
    },
    {
        "type": "server_tool_result",
        "tool_call_id": "ws_abc123",
        "status": "success"
    },
    {
        "type": "text",
        "text": "Here are some positive news stories from today...",
        "annotations": [
            {
                "end_index": 410,
                "start_index": 337,
                "title": "article title",
                "type": "citation",
                "url": "..."
            }
        ]
    }
]
See all 29 lines
This represents a single conversational turn; there are no associated ToolMessage objects that need to be passed in as in client-side tool-calling.
See the integration page for your given provider for available tools and usage details.
​
Rate limiting
Many chat model providers impose a limit on the number of invocations that can be made in a given time period. If you hit a rate limit, you will typically receive a rate limit error response from the provider, and will need to wait before making more requests.
To help manage rate limits, chat model integrations accept a rate_limiter parameter that can be provided during initialization to control the rate at which requests are made.
Initialize and use a rate limiter

​
Base URL or proxy
For many chat model integrations, you can configure the base URL for API requests, which allows you to use model providers that have OpenAI-compatible APIs or to use a proxy server.
Base URL

Proxy configuration

​
Log probabilities
Certain models can be configured to return token-level log probabilities representing the likelihood of a given token by setting the logprobs parameter when initializing the model:

Copy
model = init_chat_model(
    model="gpt-4o",
    model_provider="openai"
).bind(logprobs=True)

response = model.invoke("Why do parrots talk?")
print(response.response_metadata["logprobs"])
​
Token usage
A number of model providers return token usage information as part of the invocation response. When available, this information will be included on the AIMessage objects produced by the corresponding model. For more details, see the messages guide.
Some provider APIs, notably OpenAI and Azure OpenAI chat completions, require users opt-in to receiving token usage data in streaming contexts. See the streaming usage metadata section of the integration guide for details.
You can track aggregate token counts across models in an application using either a callback or context manager, as shown below:
Callback handler
Context manager

Copy
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="gpt-4o-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})
callback.usage_metadata

Copy
{
    'gpt-4o-mini-2024-07-18': {
        'input_tokens': 8,
        'output_tokens': 10,
        'total_tokens': 18,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}
    },
    'claude-haiku-4-5-20251001': {
        'input_tokens': 8,
        'output_tokens': 21,
        'total_tokens': 29,
        'input_token_details': {'cache_read': 0, 'cache_creation': 0}
    }
}
​
Invocation config
When invoking a model, you can pass additional configuration through the config parameter using a RunnableConfig dictionary. This provides run-time control over execution behavior, callbacks, and metadata tracking.
Common configuration options include:
Invocation with config

Copy
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",      # Custom name for this run
        "tags": ["humor", "demo"],          # Tags for categorization
        "metadata": {"user_id": "123"},     # Custom metadata
        "callbacks": [my_callback_handler], # Callback handlers
    }
)
These configuration values are particularly useful when:
Debugging with LangSmith tracing
Implementing custom logging or monitoring
Controlling resource usage in production
Tracking invocations across complex pipelines
Key configuration attributes

See full RunnableConfig reference for all supported attributes.
​
Configurable models
You can also create a runtime-configurable model by specifying configurable_fields. If you don’t specify a model value, then 'model' and 'model_provider' will be configurable by default.

Copy
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-5-nano"}},  # Run with GPT-5-Nano
)
configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
)
Configurable model with default values

Using a configurable model declaratively

Messages

Copy page

Messages are the fundamental unit of context for models in LangChain. They represent the input and output of models, carrying both the content and metadata needed to represent the state of a conversation when interacting with an LLM.
Messages are objects that contain:
 Role - Identifies the message type (e.g. system, user)
 Content - Represents the actual content of the message (like text, images, audio, documents, etc.)
 Metadata - Optional fields such as response information, message IDs, and token usage
LangChain provides a standard message type that works across all model providers, ensuring consistent behavior regardless of the model being called.
​
Basic usage
The simplest way to use messages is to create message objects and pass them to a model when invoking.

Copy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage
​
Text prompts
Text prompts are strings - ideal for straightforward generation tasks where you don’t need to retain conversation history.

Copy
response = model.invoke("Write a haiku about spring")
Use text prompts when:
You have a single, standalone request
You don’t need conversation history
You want minimal code complexity
​
Message prompts
Alternatively, you can pass in a list of messages to the model by providing a list of message objects.

Copy
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
Use message prompts when:
Managing multi-turn conversations
Working with multimodal content (images, audio, files)
Including system instructions
​
Dictionary format
You can also specify messages directly in OpenAI chat completions format.

Copy
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
​
Message types
 System message - Tells the model how to behave and provide context for interactions
 Human message - Represents user input and interactions with the model
 AI message - Responses generated by the model, including text content, tool calls, and metadata
 Tool message - Represents the outputs of tool calls
​
System Message
A SystemMessage represent an initial set of instructions that primes the model’s behavior. You can use a system message to set the tone, define the model’s role, and establish guidelines for responses.
Basic instructions

Copy
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
Detailed persona

Copy
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
​
Human Message
A HumanMessage represents user input and interactions. They can contain text, images, audio, files, and any other amount of multimodal content.
​
Text content

Message object

String shortcut

Copy
response = model.invoke([
  HumanMessage("What is machine learning?")
])
​
Message metadata
Add metadata

Copy
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)
The name field behavior varies by provider – some use it for user identification, others ignore it. To check, refer to the model provider’s reference.
​
AI Message
An AIMessage represents the output of a model invocation. They can include multimodal data, tool calls, and provider-specific metadata that you can later access.

Copy
response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>
AIMessage objects are returned by the model when calling it, which contains all of the associated metadata in the response.
Providers weigh/contextualize types of messages differently, which means it is sometimes helpful to manually create a new AIMessage object and insert it into the message history as if it came from the model.

Copy
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")

# Add to conversation history
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)
Attributes

​
Tool calls
When models make tool calls, they’re included in the AIMessage:

Copy
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    ...

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")
Other structured data, such as reasoning or citations, can also appear in message content.
​
Token usage
An AIMessage can hold token counts and other usage metadata in its usage_metadata field:

Copy
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

response = model.invoke("Hello!")
response.usage_metadata

Copy
{'input_tokens': 8,
 'output_tokens': 304,
 'total_tokens': 312,
 'input_token_details': {'audio': 0, 'cache_read': 0},
 'output_token_details': {'audio': 0, 'reasoning': 256}}
See UsageMetadata for details.
​
Streaming and chunks
During streaming, you’ll receive AIMessageChunk objects that can be combined into a full message object:

Copy
chunks = []
full_message = None
for chunk in model.stream("Hi"):
    chunks.append(chunk)
    print(chunk.text)
    full_message = chunk if full_message is None else full_message + chunk
Learn more:
Streaming tokens from chat models
Streaming tokens and/or steps from agents
​
Tool Message
For models that support tool calling, AI messages can contain tool calls. Tool messages are used to pass the results of a single tool execution back to the model.
Tools can generate ToolMessage objects directly. Below, we show a simple example. Read more in the tools guide.

Copy
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

# After a model makes a tool call
# (Here, we demonstrate manually creating the messages for brevity)
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# Execute tool and create result message
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # Must match the call ID
)

# Continue conversation
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # Model's tool call
    tool_message,  # Tool execution result
]
response = model.invoke(messages)  # Model processes the result
Attributes

The artifact field stores supplementary data that won’t be sent to the model but can be accessed programmatically. This is useful for storing raw results, debugging information, or data for downstream processing without cluttering the model’s context.
Example: Using artifact for retrieval metadata

​
Message content
You can think of a message’s content as the payload of data that gets sent to the model. Messages have a content attribute that is loosely-typed, supporting strings and lists of untyped objects (e.g., dictionaries). This allows support for provider-native structures directly in LangChain chat models, such as multimodal content and other data.
Separately, LangChain provides dedicated content types for text, reasoning, citations, multi-modal data, server-side tool calls, and other message content. See content blocks below.
LangChain chat models accept message content in the content attribute.
This may contain either:
A string
A list of content blocks in a provider-native format
A list of LangChain’s standard content blocks
See below for an example using multimodal inputs:

Copy
from langchain.messages import HumanMessage

# String content
human_message = HumanMessage("Hello, how are you?")

# Provider-native format (e.g., OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# List of standard content blocks
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
Specifying content_blocks when initializing a message will still populate message content, but provides a type-safe interface for doing so.
​
Standard content blocks
LangChain provides a standard representation for message content that works across providers.
Message objects implement a content_blocks property that will lazily parse the content attribute into a standard, type-safe representation. For example, messages generated from ChatAnthropic or ChatOpenAI will include thinking or reasoning blocks in the format of the respective provider, but can be lazily parsed into a consistent ReasoningContentBlock representation:
Anthropic
OpenAI

Copy
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
        {"type": "text", "text": "..."},
    ],
    response_metadata={"model_provider": "anthropic"}
)
message.content_blocks

Copy
[{'type': 'reasoning',
  'reasoning': '...',
  'extras': {'signature': 'WaUjzkyp...'}},
 {'type': 'text', 'text': '...'}]
See the integrations guides to get started with the inference provider of your choice.
Serializing standard content
If an application outside of LangChain needs access to the standard content block representation, you can opt-in to storing content blocks in message content.
To do this, you can set the LC_OUTPUT_VERSION environment variable to v1. Or, initialize any chat model with output_version="v1":

Copy
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano", output_version="v1")
​
Multimodal
Multimodality refers to the ability to work with data that comes in different forms, such as text, audio, images, and video. LangChain includes standard types for these data that can be used across providers.
Chat models can accept multimodal data as input and generate it as output. Below we show short examples of input messages featuring multimodal data.
Extra keys can be included top-level in the content block or nested in "extras": {"key": value}.
OpenAI and AWS Bedrock Converse, for example, require a filename for PDFs. See the provider page for your chosen model for specifics.

Image input

PDF document input

Audio input

Video input

Copy
# From URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}

# From provider-managed File ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
Not all models support all file types. Check the model provider’s reference for supported formats and size limits.
​
Content block reference
Content blocks are represented (either when creating a message or accessing the content_blocks property) as a list of typed dictionaries. Each item in the list must adhere to one of the following block types:
Core

Multimodal

Tool Calling

Server-Side Tool Execution

Provider-Specific Blocks

View the canonical type definitions in the API reference.
Content blocks were introduced as a new property on messages in LangChain v1 to standardize content formats across providers while maintaining backward compatibility with existing code.
Content blocks are not a replacement for the content property, but rather a new property that can be used to access the content of a message in a standardized format.
​
Use with chat models
Chat models accept a sequence of message objects as input and return an AIMessage as output. Interactions are often stateless, so that a simple conversational loop involves invoking a model with a growing list of messages.
Refer to the below guides to learn more:
Built-in features for persisting and managing conversation histories
Strategies for managing context windows, including trimming and summarizing messages


Tools

Copy page

Many AI applications interact with users via natural language. However, some use cases require models to interface directly with external systems—such as APIs, databases, or file systems—using structured input.
Tools are components that agents call to perform actions. They extend model capabilities by letting them interact with the world through well-defined inputs and outputs.
Tools encapsulate a callable function and its input schema. These can be passed to compatible chat models, allowing the model to decide whether to invoke a tool and with what arguments. In these scenarios, tool calling enables models to generate requests that conform to a specified input schema.
Server-side tool use
Some chat models (e.g., OpenAI, Anthropic, and Gemini) feature built-in tools that are executed server-side, such as web search and code interpreters. Refer to the provider overview to learn how to access these tools with your specific chat model.
​
Create tools
​
Basic tool definition
The simplest way to create a tool is with the @tool decorator. By default, the function’s docstring becomes the tool’s description that helps the model understand when to use it:

Copy
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
Type hints are required as they define the tool’s input schema. The docstring should be informative and concise to help the model understand the tool’s purpose.
​
Customize tool properties
​
Custom tool name
By default, the tool name comes from the function name. Override it when you need something more descriptive:

Copy
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search
​
Custom tool description
Override the auto-generated tool description for clearer model guidance:

Copy
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
​
Advanced schema definition
Define complex inputs with Pydantic models or JSON schemas:

Pydantic model

JSON Schema

Copy
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
​
Reserved argument names
The following parameter names are reserved and cannot be used as tool arguments. Using these names will cause runtime errors.
Parameter name	Purpose
config	Reserved for passing RunnableConfig to tools internally
runtime	Reserved for ToolRuntime parameter (accessing state, context, store)
To access runtime information, use the ToolRuntime parameter instead of naming your own arguments config or runtime.
​
Accessing Context
Why this matters: Tools are most powerful when they can access agent state, runtime context, and long-term memory. This enables tools to make context-aware decisions, personalize responses, and maintain information across conversations.
Runtime context provides a way to inject dependencies (like database connections, user IDs, or configuration) into your tools at runtime, making them more testable and reusable.
Tools can access runtime information through the ToolRuntime parameter, which provides:
State - Mutable data that flows through execution (e.g., messages, counters, custom fields)
Context - Immutable configuration like user IDs, session details, or application-specific configuration
Store - Persistent long-term memory across conversations
Stream Writer - Stream custom updates as tools execute
Config - RunnableConfig for the execution
Tool Call ID - ID of the current tool call
⚡ Enhanced Tool Capabilities

📊 Available Resources

🔧 Tool Runtime Context

Tool Call

ToolRuntime

State Access

Context Access

Store Access

Stream Writer

Messages

Custom State

User ID

Session Info

Long-term Memory

User Preferences

Context-Aware Tools

Stateful Tools

Memory-Enabled Tools

Streaming Tools

​
ToolRuntime
Use ToolRuntime to access all runtime information in a single parameter. Simply add runtime: ToolRuntime to your tool signature, and it will be automatically injected without being exposed to the LLM.
ToolRuntime: A unified parameter that provides tools access to state, context, store, streaming, config, and tool call ID. This replaces the older pattern of using separate InjectedState, InjectedStore, get_runtime, and InjectedToolCallId annotations.
The runtime automatically provides these capabilities to your tool functions without you having to pass them explicitly or use global state.
Accessing state:
Tools can access the current graph state using ToolRuntime:

Copy
from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
The runtime parameter is hidden from the model. For the example above, the model only sees pref_name in the tool schema - runtime is not included in the request.
Updating state:
Use Command to update the agent’s state or control the graph’s execution flow:

Copy
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})
​
Context
Access immutable configuration and contextual data like user IDs, session details, or application-specific configuration through runtime.context.
Tools can access runtime context through ToolRuntime:

Copy
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)
​
Memory (Store)
Access persistent data across conversations using the store. The store is accessed via runtime.store and allows you to save and retrieve user-specific or application-specific data.
Tools can access and update the store through ToolRuntime:

Copy
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev
See all 42 lines
​
Stream Writer
Stream custom updates from tools as they execute using runtime.stream_writer. This is useful for providing real-time feedback to users about what a tool is doing.

Copy
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"



    Short-term memory

Copy page

​
Overview
Memory is a system that remembers information about previous interactions. For AI agents, memory is crucial because it lets them remember previous interactions, learn from feedback, and adapt to user preferences. As agents tackle more complex tasks with numerous user interactions, this capability becomes essential for both efficiency and user satisfaction.
Short term memory lets your application remember previous interactions within a single thread or conversation.
A thread organizes multiple interactions in a session, similar to the way email groups messages in a single conversation.
Conversation history is the most common form of short-term memory. Long conversations pose a challenge to today’s LLMs; a full history may not fit inside an LLM’s context window, resulting in an context loss or errors.
Even if your model supports the full context length, most LLMs still perform poorly over long contexts. They get “distracted” by stale or off-topic content, all while suffering from slower response times and higher costs.
Chat models accept context using messages, which include instructions (a system message) and inputs (human messages). In chat applications, messages alternate between human inputs and model responses, resulting in a list of messages that grows longer over time. Because context windows are limited, many applications can benefit from using techniques to remove or “forget” stale information.
​
Usage
To add short-term memory (thread-level persistence) to an agent, you need to specify a checkpointer when creating an agent.
LangChain’s agent manages short-term memory as a part of your agent’s state.
By storing these in the graph’s state, the agent can access the full context for a given conversation while maintaining separation between different threads.
State is persisted to a database (or memory) using a checkpointer so the thread can be resumed at any time.
Short-term memory updates when the agent is invoked or a step (like a tool call) is completed, and the state is read at the start of each step.

Copy
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  


agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  
)
​
In production
In production, use a checkpointer backed by a database:

Copy
pip install langgraph-checkpoint-postgres

Copy
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver  


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,  
    )
​
Customizing agent memory
By default, agents use AgentState to manage short term memory, specifically the conversation history via a messages key.
You can extend AgentState to add additional fields. Custom state schemas are passed to create_agent using the state_schema parameter.

Copy
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",  
        "preferences": {"theme": "dark"}  
    },
    {"configurable": {"thread_id": "1"}})
​
Common patterns
With short-term memory enabled, long conversations can exceed the LLM’s context window. Common solutions are:
Trim messages
Remove first or last N messages (before calling LLM)
Delete messages
Delete messages from LangGraph state permanently
Summarize messages
Summarize earlier messages in the history and replace them with a summary
Custom strategies
Custom strategies (e.g., message filtering, etc.)
This allows the agent to keep track of the conversation without exceeding the LLM’s context window.
​
Trim messages
Most LLMs have a maximum supported context window (denominated in tokens).
One way to decide when to truncate messages is to count the tokens in the message history and truncate whenever it approaches that limit. If you’re using LangChain, you can use the trim messages utility and specify the number of tokens to keep from the list, as well as the strategy (e.g., keep the last max_tokens) to use for handling the boundary.
To trim message history in an agent, use the @before_model middleware decorator:

Copy
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
​
Delete messages
You can delete messages from the graph state to manage the message history.
This is useful when you want to remove specific messages or clear the entire message history.
To delete messages from the graph state, you can use the RemoveMessage.
For RemoveMessage to work, you need to use a state key with add_messages reducer.
The default AgentState provides this.
To remove specific messages:

Copy
from langchain.messages import RemoveMessage  

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}  
To remove all messages:

Copy
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}  
When deleting messages, make sure that the resulting message history is valid. Check the limitations of the LLM provider you’re using. For example:
Some providers expect message history to start with a user message
Most providers require assistant messages with tool calls to be followed by corresponding tool result messages.

Copy
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

Copy
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.')]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
[('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
​
Summarize messages
The problem with trimming or removing messages, as shown above, is that you may lose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model.

To summarize message history in an agent, use the built-in SummarizationMiddleware:

Copy
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob!
"""
See SummarizationMiddleware for more configuration options.
​
Access memory
You can access and modify the short-term memory (state) of an agent in several ways:
​
Tools
​
Read short-term memory in a tool
Access short term memory (state) in a tool using the ToolRuntime parameter.
The tool_runtime parameter is hidden from the tool signature (so the model doesn’t see it), but the tool can access the state through it.

Copy
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.
​
Write short-term memory from tools
To modify the agent’s short-term memory (state) during execution, you can return state updates directly from the tools.
This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.

Copy
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):  
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={  
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState, 
    context_schema=CustomContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
​
Prompt
Access short term memory (state) in middleware to create dynamic prompts based on conversation history or custom state fields.

Copy
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class CustomContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()

Output

Copy
================================ Human Message =================================

What is the weather in SF?
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_WFQlOGn4b2yoJrv7cih342FG)
 Call ID: call_WFQlOGn4b2yoJrv7cih342FG
  Args:
    city: San Francisco
================================= Tool Message =================================
Name: get_weather

The weather in San Francisco is always sunny!
================================== Ai Message ==================================

Hi John Smith, the weather in San Francisco is always sunny!
​
Before model
Access short term memory (state) in @before_model middleware to process messages before model calls.
__start__

before_model

model

tools

__end__


Copy
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
​
After model
Access short term memory (state) in @after_model middleware to process messages after model calls.
__start__

model

after_model

tools

__end__


Copy
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)


Filter responses based on the documents that allow end-user permissions.
Amazon also offers options for companies that want to develop more customized generative AI solutions. Amazon SageMaker JumpStart is a hub for machine learning with base models, built-in algorithms, and pre-built machine learning solutions that you can deploy with just a few clicks. You can accelerate RAG implementation by referencing existing SageMaker notebooks and code
