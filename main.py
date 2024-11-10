from tqdm import tqdm
import toml
import json
from pathlib import Path


from langchain_core.runnables import RunnableSequence
from argparse import ArgumentParser

from utils import extract_metadata, get_first_page_image, prepare_chains, load_pdf
from pydantic import ValidationError


###################################################


def load_prompts(toml_path: str) -> dict:
    with open(toml_path, "r") as f:
        prompts = toml.load(f)

    return prompts


def extract_essentials(text: str, chain: RunnableSequence) -> dict:
    try:
        essentials = chain.invoke({"text": text}).model_dump()
    except ValidationError:
        raise ValueError(f"Failed to parse\n{essentials}")
    except Exception as e:
        raise Exception(f"Error extracting essentials\n{essentials}\n{e}")
    
    return essentials


# def extract_section_titles(text: str, chain: RunnableSequence) -> list[str]:
#     section_titles = chain.invoke({"text": text}).split("\n")
#     return section_titles


# def extract_section_details(text: str, section_titles: list[str], chain: RunnableSequence) -> list[str]:
#     section_details = []
#     for section_title in tqdm(section_titles):
#         section_detail = chain.invoke({
#             "text": text,
#             "section_title": section_title,
#         })
#         section_details.append(f"## {section_title}\n\n{section_detail}")

#     return section_details

# def extract_toc(text: str, chain: RunnableSequence) -> str:
#     toc = chain.invoke({"text": text})
#     return toc

def extract_section_summary(text: str, chain: RunnableSequence) -> str:
    section_summary = chain.invoke({"text": text})
    return section_summary

def get_overview(pdf_path: str, toml_path: str, write_to_file: bool = True) -> str:
    """
    Gets the overview of a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        toml_path (str): The path to the prompts toml file.
        write_to_file (bool): If True, writes the output to a file as the same name as the PDF file but with a .md extension.

    Returns:
        str: The overview of the PDF file.
    """
    temp_path = Path(pdf_path)
    output_dir = temp_path.parent
    if temp_path.exists() and temp_path.suffix == ".pdf":
        file_name = temp_path.stem
    else:
        raise FileNotFoundError(f"File {pdf_path} does not exist or is not a PDF file")

    prompts = load_prompts(toml_path)
    essentials_chain = prepare_chains(prompts, "essentials")
        
    first_page_image = get_first_page_image(pdf_path)
    md_text = load_pdf(pdf_path)

    print("Extracting metadata...\n")
    metadata = extract_metadata(first_page_image)
    
    print("Extracting essentials...\n")
    essentials = extract_essentials(md_text, essentials_chain)
    tldr, key_takeaways, importance_to_researchers, toc = essentials.get("tldr", ""), essentials.get("key_takeaways", []), essentials.get("importance_to_researchers", ""), essentials.get("toc", "")
    
    section_summary_chain = prepare_chains(prompts, "section_summary", payload=toc)
    print("Extracting section summary...\n")
    section_summary = extract_section_summary(md_text, section_summary_chain)
    
    all_output = ""

    all_output += f"# {metadata.get('title', '')}\n\n"
    all_output += f"#### {', '.join(metadata.get('authors', []))}\n\n"
    all_output += f"##### {metadata.get('affiliation', '')}\n\n"
    all_output += "### TL;DR\n\n"
    all_output += f"{tldr}\n\n"
    all_output += "### Key Takeaways\n\n"
    for key in key_takeaways:
        all_output += f"- {key}\n\n"
    all_output += "### Importance to Researchers\n\n"
    all_output += f"{importance_to_researchers}\n\n"
    all_output += "### Abstract\n\n"
    for paragraph in metadata.get("abstract", "").split("\n"):
        all_output += f">{paragraph}\n\n"
    all_output += "### Table of Contents\n\n"
    all_output += f"{toc}\n\n"
    all_output += "- - -\n\n"
    all_output += f"{section_summary}\n\n"
    if write_to_file:
        with open(f"{output_dir}/{file_name}_overview.md", "w", encoding="utf-8") as f:
            f.write(all_output)
            print(f"File {file_name}_overview.md written successfully\n")

    return all_output


def process_dir(dir_path: str, prompts_path: str, write_to_file: bool = True):
    """
    Processes all PDF files in a directory.

    Args:
        dir_path (str): The path to the directory.
        prompts_path (str): The path to the prompts toml file.
        write_to_file (bool): If True, writes the output to a file as the same name as the PDF file but with a .md extension.
    """
    input_dir_path = Path(dir_path)
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        raise FileNotFoundError(f"Directory {dir_path} does not exist or is not a directory")
    files = input_dir_path.glob("*.pdf")
    # if len(list(files)) == 0:
    #     raise FileNotFoundError(f"No PDF files found in directory {dir_path}")
    try:
        for file in tqdm(files):
            print(f"Processing {file}...")
            get_overview(file, prompts_path, write_to_file)
    except Exception as e:
        print(f"Error processing {file}:\n{e}")


if __name__ == "__main__":
    parser = ArgumentParser("Extracts the overview of a PDF file or a directory of PDF files")
    parser.add_argument("pdf_path", nargs="?", type=str, help="Path to the PDF file")
    parser.add_argument("--dir", "-d", nargs="?", help="Process all PDF files in the directory", default=False)
    parser.add_argument("--prompts_path", "-p", type=str, help="Path to the prompts toml file", required=True)
    parser.add_argument("--write_to_file", "-w", action="store_true", help="Write the output to a file", default=True)
    args = parser.parse_args()
    if args.dir:
        process_dir(args.dir, args.prompts_path, args.write_to_file)
    else:
        get_overview(args.pdf_path, args.prompts_path, args.write_to_file)
