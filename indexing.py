from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from filereader import read_multiple_pdfs

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectore_store = InMemoryVectorStore(embeddings)

DOC_PATHS = [
    "pdfs/climate_change.pdf",
    "pdfs/global_warming.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)

all_splits = text_splitter.split_documents(docs)

print(f"split dpcumentation into {len(all_splits)} chunks")

vectore_store.add_documents(documents=all_splits)
print(f"indexed {len(all_splits)} chunks")


