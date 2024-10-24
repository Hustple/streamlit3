#vector store initialization
from pinecone import Pinecone, ServerlessSpec
import time
# from config import VECTOR_DIMENSION
VECTOR_DIMENSION = 384 # 768 do if nomic embeddings using
class PineconeManager:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None
        self.initialize_index() #to initialize the index
    def initialize_index(self):
        if self.index_name not in self.pc.list_indexes().names():#shd use list indexes here
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            print(f"Index {self.index_name} already exists")

        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        self.index = self.pc.Index(self.index_name)