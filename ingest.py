import os
import shutil
import csv
import pickle

#from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS





def process_and_delete_csv(file_path, vectorstore):
    loader = UnstructuredFileLoader(file_path=file_path)
    data = loader.load()

    embeddings = OpenAIEmbeddings()
    vectorstore =  FAISS.from_documents(data, embeddings)

    os.remove(file_path)


def ingest_docs():

    vectorstore = None

    txt_folder = "./txt"
    for file_name in os.listdir(txt_folder):
        file_path = os.path.join(txt_folder, file_name)
        loader = UnstructuredFileLoader(file_path=file_path)
        data = loader.load()

        embeddings = OpenAIEmbeddings()
        if vectorstore is None:
            vectorstore = FAISS.from_documents(data, embeddings)
        else:
            new_vectorstore = FAISS.from_documents(data, embeddings)
            vectorstore.merge_from(new_vectorstore)

        os.remove(file_path)

    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_docs()



