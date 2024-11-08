from tqdm import tqdm
import toml
from pathlib import Path


from langchain_core.runnables import RunnableSequence
from argparse import ArgumentParser

from utils import extract_metadata, get_first_page_image, convert_to_markdown, prepare_chains


###################################################


def load_prompts(toml_path: str) -> dict:
    with open(toml_path, "r") as f:
        prompts = toml.load(f)

    return prompts


def extract_essentials(text: str, chain: RunnableSequence) -> str:
    essentials = chain.invoke({"text": text})
    return essentials


def extract_section_titles(text: str, chain: RunnableSequence) -> list[str]:
    section_titles = chain.invoke({"text": text}).split("\n")
    return section_titles


def extract_section_details(text: str, section_titles: list[str], chain: RunnableSequence) -> list[str]:
    section_details = []
    for section_title in tqdm(section_titles):
        section_detail = chain.invoke({
            "text": text,
            "section_title": section_title,
        })
        section_details.append(f"## {section_title}\n\n{section_detail}")

    return section_details


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
    sections_chain = prepare_chains(prompts, "sections")
    section_details_chain = prepare_chains(prompts, "section_details")

    first_page_image = get_first_page_image(pdf_path)
    md_text = convert_to_markdown(pdf_path, remove_references=True)
    print("Extracting metadata...\n")
    metadata = extract_metadata(first_page_image)
    print("Extracting essentials...\n")
    essentials = extract_essentials(md_text, essentials_chain)
    print("Extracting section titles...\n")
    section_titles = extract_section_titles(md_text, sections_chain)
    print("Extracting section details...\n")
    section_details = extract_section_details(md_text, section_titles, section_details_chain)

    all_output = ""

    all_output += f"# {metadata['title']}\n\n"
    all_output += f"### {', '.join(metadata['authors'])}\n\n"
    all_output += f"#### {metadata['affiliation']}\n\n"
    all_output += "- - -\n\n"
    all_output += f"{essentials}\n\n"
    all_output += "- - -\n\n"
    all_output += "## Abstract\n\n"
    all_output += f"{metadata['abstract']}\n\n"
    all_output += "- - -\n\n"
    all_output += "\n\n".join(section_details)
    if write_to_file:
        with open(f"{output_dir}/{file_name}_overview.md", "w", encoding="utf-8") as f:
            f.write(all_output)
            print(f"File {file_name}.md written successfully\n")

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
    if len(files) == 0:
        raise FileNotFoundError(f"No PDF files found in directory {dir_path}")

    for file in tqdm(files):
        get_overview(file, prompts_path, write_to_file)


if __name__ == "__main__":
    parser = ArgumentParser("Extracts the overview of a PDF file or a directory of PDF files")
    parser.add_argument("pdf_path", nargs="?", type=str, help="Path to the PDF file")
    parser.add_argument("--dir", "-d", nargs="?", help="Process all PDF files in the directory", default=False)
    parser.add_argument("--prompts_path", "-p", type=str, help="Path to the prompts toml file", required=True)
    parser.add_argument("--write_to_file", "-w", action="store_true", help="Write the output to a file", default=True)
    args = parser.parse_args()
    if args.dir:
        process_dir(args.pdf_path, args.prompts_path, args.write_to_file)
    else:
        get_overview(args.pdf_path, args.prompts_path, args.write_to_file)
