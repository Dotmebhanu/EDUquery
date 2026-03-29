from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import tempfile,os

#Main Agenda: loading the user uploded files into langchain_loader through langchain documents
"""langchain documents stores :
                               1.page_content :text or original file content
                               2.meta_data where where we store filename in this case"""
def load_document(file_bytes:bytes , filename:str):
    suffix=os.path.splitext(filename)[1].lower()

#we first store the uploaded data into one temp file through this tempfile we load our langchain becoz langchain needs just path to load the. file
# and the "tempfile" generates automatic temp file in os and we extract path of it to load for langchains
    
    with tempfile.NamedTemporaryFile(delete=False , suffix=suffix) as temp:
        temp.write(file_bytes)
        temp_path=temp.name
    try:
            if suffix==".pdf":
                loader=PyPDFLoader(temp_path)
            elif suffix==".docx":
                loader=Docx2txtLoader(temp_path)
            elif suffix in  [".ppt",".pptx"]:
                loader=UnstructuredPowerPointLoader(temp_path)
            else:
                raise ValueError(f"unsupporteed file type : {suffix}")
            documents =loader.load()

            #page content is done 

            #now meta data to be stored , here for each  file we store filename as metadata/refernce

            for doc in documents:
                doc.metadata["filename"]=filename

            return documents
    finally:
        os.unlink(temp_path) #unlinking the temp data path
                

