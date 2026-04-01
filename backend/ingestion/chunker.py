from langchain_text_splitters import RecursiveCharacterTextSplitter

"""splitting the doc into chunks and adding chunk index to metadata for citation"""
def chunk_documents(documents):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )

    chunks=splitter.split_documents(documents)

    #giving each chunk an index for citations

    for i,chunk in  enumerate(chunks):
        chunk.metadata["chunk_index"]=i

    return chunks 

