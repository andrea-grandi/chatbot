from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv


load_dotenv()
loader = DirectoryLoader('data', glob='*.md')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len,
    add_start_index=True,
)
chunks = splitter.split_documents(docs)
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embedding)
db.save_local("faiss_index")
