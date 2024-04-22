import os
import sys
import const
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = const.APIKEY

query = sys.argv[1]

loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))