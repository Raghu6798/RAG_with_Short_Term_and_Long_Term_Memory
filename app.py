import gradio as gr
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.vectorstores import Neo4jVector
from langchain_google_genai import ChatGoogleGenerativeAI
from uuid import uuid4
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Initialize variables
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

# Neo4j graph setup
graph = Neo4jGraph(
    url="neo4j+s://6682e6ce.databases.neo4j.io",
    username="neo4j",
    password=os.getenv("NEO4J_PASSWORD")
)

# HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Create Neo4j VectorStore
graph_store = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="vector",
    embedding_node_property="Embedding",
    text_node_property="text",
    retrieval_query="""
// get the document
MATCH (node)-[:PART_OF]->(d:Document)
WITH node, score, d

// get the entities and relationships for the document
MATCH (node)-[:HAS_ENTITY]->(e)
MATCH p = (e)-[r]-(e2)
WHERE (node)-[:HAS_ENTITY]->(e2)

// unwind the path, create a string of the entities and relationships
UNWIND relationships(p) as rels
WITH
    node,
    score,
    d,
    collect(apoc.text.join(
        [labels(startNode(rels))[0], startNode(rels).id, type(rels), labels(endNode(rels))[0], endNode(rels).id]
        ," ")) as kg
RETURN
    node.text as text, score,
    {
        document: d.id,
        entities: kg
    } AS metadata
""")
retriever = graph_store.as_retriever()

# Define Cypher Prompt
CYPHER_PROMPT = """
(
    "Use the given context to provide an in-depth and structured response."
    "Your answer should include:"
    "- A clear and concise introduction to the topic."
    "- Detailed explanation or relevant steps to address the query."
    "- Practical examples or applications where possible."
    "- A conclusion summarizing the main points."
    "Format your response in sections with appropriate headings for clarity."
    "Context: {context}"
)
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", CYPHER_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Helper function to retrieve context
def get_retrieved_context(query: str) -> str:
    retrieved_documents = retriever.get_relevant_documents(query)
    context = "\n".join(doc.page_content for doc in retrieved_documents)
    return context

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

def ReturnResponse(query: str) -> str:
    llm =  ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    )
    chat_chain = prompt | llm | StrOutputParser()

    chat_with_message_history = RunnableWithMessageHistory(
        chat_chain,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    context = get_retrieved_context(query)
    response = chat_with_message_history.invoke({
        "question": query,
        "context": context,
    }, config={
        "configurable": {"session_id": SESSION_ID}
    })

    return gr.Markdown(response)


iface = gr.Interface(
    fn=ReturnResponse,
    inputs=gr.Textbox(label="Enter your query:", placeholder="Type your question here..."),
    outputs=gr.Markdown(label="Chatbot Response"),
    title="GraphRAG with conversational Memory 🤖💬"
)

iface.launch()
