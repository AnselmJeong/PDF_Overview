# %%
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path

load_dotenv()

def load_pdf(pdf_path: str) -> str:
    """
    Loads a PDF file and returns the total content as a string.
    args:
        pdf_path (str): The path to the PDF file.
    returns:
        str: The total content of the PDF file as a string.
    """
    temp_path = Path(pdf_path)
    if not temp_path.exists() or not temp_path.suffix == ".pdf":
        raise FileNotFoundError(f"The temporary file {temp_path} does not exist or is not a PDF file.")
    else:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        total_content = "\n\n".join([doc.page_content for doc in docs])
        
    return total_content

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                             api_key=os.getenv("GOOGLE_API_KEY"), 
                             max_tokens=1048576)


toc_prompt = ChatPromptTemplate.from_messages([
    ("system", """Create a table of contents from the provided text. The full text is divided into several sections, with each section's first line containing a section heading or subheading. The font size of the (sub)headings determines the hierarchy of the sections. After determining the importance of each section based on the hierarchy, extract not more than 10 major sections and create a corresponding table of contents. Ignore references, footnotes, and other non-content text.
    """),
    ("user", "{text}"),
])

toc_chain = toc_prompt | llm

toc = toc_chain.invoke({"text": total_content}).content

toc

# %%
summary_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """The provided text is composed of several sections and sub-sections structured by the following table of contents.
     {toc}
     Analyze and generate comprehensive thoughts about the content of each sub-section structured by the table of contents one by one consecutively.
     - No subsection should be left out in the output.
     - The thoughts for each sub-section should be written with a thoughtful and in-depth approach to uncover valuable insights.
     - The thoughts for each sub-section should be written in a single paragraph, do not use bullet points, and it should be at least 1500 characters long but not more than 2000 characters.
     - Use **bold** formatting to emphasize key points.
    """),
    ("user", "{text}"),
])

summary_template = summary_prompt_template.partial(toc=toc)

summary_chain = summary_template | llm

summary = summary_chain.invoke({"text": total_content}).content


# %%
from IPython.display import Markdown

Markdown(summary)

# %%
summary

# %%



