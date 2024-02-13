import streamlit as st
import openai
import qdrant_client
import os
from langchain.vectorstores import Qdrant

# used to create the memory
from langchain.memory import ConversationBufferMemory

# used to load text
from langchain.document_loaders import WebBaseLoader

# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings

# used to create the retrieval tool
from langchain.agents import tool

# used to create the prompt template
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder


# used to create the agent executor
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor


# set the secure key
openai_api_key = st.secrets.openai_key
QDRANT_HOST = st.secrets.QDRANT_HOST
QDRANT_API_KEY = st.secrets.QDRANT_API_KEY
os.environ["COHERE_API_KEY"] = st.secrets.COHERE_API_KEY

# add a heading for your app.
st.header("Chat with the GSM mall ")

# Initialize the memory
# This is needed for both the memory and the prompt
memory_key = "history"

if "memory" not in st.session_state.keys():
    st.session_state.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Need Info? Ask Me Questions about GSM Mall's Features"}
    ]

# create the document database
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the LLM blog – hang tight!."):
        client = qdrant_client.QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )
        embeddings = CohereEmbeddings(model="embed-english-v2.0")
        vector_store = Qdrant(
            client=client,
            collection_name="gsm_demo02",
            embeddings=embeddings
        )
        print("connection established !")
        return vector_store 

db = load_data()


# instantiate the database retriever
retriever = db.as_retriever(search_type="mmr")

# define the retriever tool
@tool
def tool(query):
    "Searches and returns documents regarding the llm powered autonomous agents blog"
    docs = retriever.get_relevant_documents(query)
    return docs

tools = [tool]

# define the prompt
system_message = SystemMessage(
        content=(
            "Your friendly assistant is here to help! Remember, always provide clear, concise, and friendly responses within 10-15 words. value User time and aim to provide clear and concise responses. Maintain a positive and professional tone. Encourage users to visit the store subtly, without being pushy. Dont hallucinate. Let's make every interaction a delightful experience! 😊"
        )
)
prompt_template = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

# instantiate the large language model
llm = ChatOpenAI(temperature = 0, openai_api_key=openai_api_key)

# instantiate agent
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

# Prompt for user input and display message history
if prompt := st.chat_input("Your LLM based agent related question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor({"input": prompt})
            st.write(response["output"])
            message = {"role": "assistant", "content": response["output"]}
            st.session_state.messages.append(message) # Add response to message history
