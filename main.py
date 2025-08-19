from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chat_models import init_chat_model


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


file_path = "C:\\Users\\aravi\\Downloads\\archive (1)\\FINAL FOOD DATASET\\FOOD-DATA-GROUP1.csv"

loader = CSVLoader(file_path=file_path)
data = loader.load()

for record in data[:1]:
    print(record)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200, 
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(data)

print("************************************************")
print(f"Split data into {len(all_splits)} sub-documents.")


document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)


from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



result = graph.invoke({"question": "what is the nutrition density of  cream cheese?"})

print(f"Context: {result['context']}\n\n")
print(f"Answer: {result['answer']}")