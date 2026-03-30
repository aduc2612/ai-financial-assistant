import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings

df = pd.read_csv("archive/financial_news_events.csv")
df = df.drop(columns = ["News_Url"])
df = df[df['Headline'].notna()]
df = df[df['Index_Change_Percent'].notna()]
df = df[df['Sentiment'].notna()]

embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for _, row in df.iterrows():
        id = str(uuid4())
        document = Document(
            page_content = row["Headline"] + ", impact: " + row["Impact_Level"],
            metadata = {
                "date": row["Date"], 
                "source": row["Source"], 
                "event_type": row["Market_Event"], 
                "affected_market_index": row["Market_Index"],
                "market_index_percent_change": row["Index_Change_Percent"],
                "trading_volume": row["Trading_Volume"],
                "sentiment": row["Sentiment"],
                "sector": row["Sector"],
                "related_company": row["Related_Company"]
            },
            id = id
        )
        ids.append(id)
        documents.append(document)

vector_store = Chroma(
    collection_name="financial_reports",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents = documents, ids = ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)