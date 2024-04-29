from pymilvus import MilvusClient

vector_db = MilvusClient(
    uri="http://localhost:19530"
)

# CHANGE "COLLECTION_NAME" ARGUMENT WHEN DELETING AN EXISTING CONTENT #
res = vector_db.drop_collection(
    collection_name=""
)

print(res)
