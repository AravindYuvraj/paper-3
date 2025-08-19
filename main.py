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

