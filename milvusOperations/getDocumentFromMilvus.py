import config
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# CHANGE "COLLECTION_NAME" ARGUMENT TO SAVING NEW CONTENT #
vector_db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="placeholder",
)

# Change query to get relevant information #
query = "Hacizden Sonra Taksitle Ödeme"
docs = vector_db.similarity_search(query)


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


print(format_docs(docs))
