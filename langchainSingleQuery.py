import config
from langchain import hub
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# Setup
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("ahmet-kaan-ozbek/icra_iflas_danismani")

# Collection names and QUERY which is the QUESTION
collection_names = ["icra_iflas_not_1"]
query = ("Bir satır üstte paylaşılan somut olayda, ihtiyati haciz kararı hangi yer veya yerler mahkemesinden "
         "alınabilir? Anlatınız.")
case = ("İstanbul’da Maslak-Levent’te yapı ve tasarım işleriyle uğraşan Batuhan Büyükusta firmasında kullanılmak üzere "
        "01 Eylül 2016 tarihinde merkezi Kadıköy’de bulunan VIEN&VIENNA Mobilyacılık Tic. Ltd. Şti’den 75.000 TL "
        "tutarında mobilya satın almış ve karşılığında da her ay ödenmek üzere 25.000 TL olmak üzere 3 adet bono "
        "vermiştir. Taraflar arasında yapılan sözleşme hükümlerine göre, ilk taksit bedelinin vade günü, "
        "01 Kasım 2016 olarak kararlaştırılmıştır. Vade günü gelmesine rağmen borcu ilk taksiti zamanında "
        "ödeyememiştir. Alacaklı şirket yetkilisi, borçluyu müteaddit defalar aramasına rağmen borçlunun "
        "telefonlarına cevap vermediğini ve ofisinin de üç aydır kapalı olduğunu tespit etmiştir. Bu durumda "
        "alacaklarını tahsil edemeyecekleri endişesiyle karşı karşıya kalan şirketin vekili Av. Ali Akarca borcu "
        "aleyhine ihtiyati haciz kararı almayı düşünmektedir.")

all_formatted_docs = ""

# Processing and concatenating results of each collection
for collection_name in collection_names:
    vector_db = Milvus(
        embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name=collection_name
    )
    retriever_results = vector_db.similarity_search(query)
    formatted_docs = format_docs(retriever_results)
    all_formatted_docs += formatted_docs + "\n"

# Execute the RAG chain
rag_chain = (
        {"context": lambda x: all_formatted_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print(rag_chain.invoke(case + "\n" + query))
