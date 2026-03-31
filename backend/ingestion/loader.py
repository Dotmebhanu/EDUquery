import pdfplumber
import tempfile
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

"""
Main Agenda: loading user uploaded files into LangChain Documents
LangChain Document stores:
    1. page_content — text content of the file
    2. metadata     — filename, page number for citations
"""

def load_document(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1].lower()

    # Save bytes to temp file — LangChain loaders need a file path not raw bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    try:
        if suffix == ".pdf":
            documents = []  # ✅ outside the loop — not inside

            with pdfplumber.open(temp_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract plain text
                    text = page.extract_text() or ""

                    # Extract tables and convert to readable text
                    tables = page.extract_tables() or []
                    table_text = ""
                    for table in tables:
                        for row in table:
                            # Handle None cells — pdfplumber returns None for empty cells
                            row_text = " | ".join(
                                cell if cell is not None else ""
                                for cell in row
                            )
                            table_text += row_text + "\n"

                    # Combine paragraph text and table text
                    full_text = text + "\n" + table_text

                    documents.append(Document(
                        page_content=full_text,
                        metadata={
                            "filename": filename,
                            "page": page_num + 1  # humans count pages from 1
                        }
                    ))

            return documents  # ✅ return here — don't fall through to loader.load()

        elif suffix == ".docx":
            loader = Docx2txtLoader(temp_path)

        elif suffix in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(temp_path)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # DOCX and PPTX reach here — PDF returned early above
        documents = loader.load()

        # Add filename to metadata for citations
        for doc in documents:
            doc.metadata["filename"] = filename

        return documents

    finally:
        os.unlink(temp_path)  # Always delete temp file even if error occurs