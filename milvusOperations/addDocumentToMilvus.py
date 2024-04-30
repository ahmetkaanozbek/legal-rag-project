import config
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# CHANGE DOCUMENT PATH TO ADD NEW EMBEDDINGS TO THE VECTOR STORE #
loader = TextLoader("/Users/kaanozbek/Downloads/turk_icra_iflas_kanunu.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# CHANGE "COLLECTION_NAME" ARGUMENT WHEN SAVING A NEW DOCUMENT #
vector_db = Milvus.from_documents(
    docs,
    embeddings,
    collection_name="icra_iflas_kanunu",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

for doc in docs:
    print(doc.page_content)
    print("\n")
