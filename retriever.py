from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class Retriever:

    def __init__(
            self,
            index_path,
            embedding_function,
            top_k=5):

        self.index_path = index_path
        self.embedding_function = embedding_function

        self.db = FAISS.load_local(
            index_path,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": top_k})

    def retrive(self, query):
        return self.retriever.invoke(query)
