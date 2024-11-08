import pymupdf
from pymupdf4llm import to_markdown
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

MODEL = "grok-beta"
VISION_MODEL = "llama-3.2-11b-vision-preview"
BASE_URL = "https://api.x.ai/v1"
load_dotenv()

llm_model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=os.getenv("XAI_API_KEY"))


def get_first_page_image(pdf_path: str) -> bytes:
    """
    Extracts the first page image from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        bytes: The image of the first page in bytes.
    """
    pdf_document = pymupdf.open(pdf_path)
    pixmap = pdf_document[0].get_pixmap()

    return pixmap.tobytes()


def convert_to_markdown(pdf_path: str, write_to_file: bool = False, remove_references: bool = False) -> str:
    """
    Converts a PDF file to markdown format.

    Args:
        pdf_path (str): The path to the PDF file.
        write_to_file (bool): If True, writes the markdown text to a file as the same name as the PDF file but with a .md extension.

    Returns:
        str: The markdown text of the PDF.
    """
    temp_path = Path(pdf_path)
    output_dir = temp_path.parent
    if temp_path.exists() and temp_path.suffix == ".pdf":
        file_name = temp_path.stem
    else:
        raise FileNotFoundError(f"File {pdf_path} does not exist or is not a PDF file")

    md_text = to_markdown(pdf_path)
    if write_to_file:
        with open(f"{output_dir}/{file_name}.md", "w", encoding="utf-8") as f:
            f.write(md_text)

    if remove_references:
        md_text = md_text[: md_text.lower().find("references")]

    return md_text


def prepare_chains(prompts: dict, purpose: str) -> RunnableSequence:
    """
    Prepare the chains for the extraction of the essentials, sections and section details.

    Args:
        prompts (dict): The prompts to use for the extraction.
        purpose (str): The purpose of the extraction. Can be "essentials", "sections" or "section_details".

    Returns:
        RunnableSequence: The chain for the extraction of the essentials, sections and section details.
    """
    prompt_template = prompts[purpose]["prompt"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("user", "{text}"),
    ])

    chain = prompt | llm_model | StrOutputParser()

    return chain
