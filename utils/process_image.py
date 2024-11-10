import ollama
import json
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


VISION_MODEL = "llama3.2-vision"
MODEL = "aya-expanse:32b"



class Metadata(BaseModel):
    title: str = Field(description="The title of the article")
    authors: list[str] = Field(description="The list of authors of the article")
    affiliation: str = Field(description="The affiliation of the first author")
    abstract: str = Field(description="The abstract of the article")

llm = ChatOllama(model=MODEL).with_structured_output(Metadata)
    
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that extract structured JSON from the input by the user."),
    ("user", "{input}"),
])

chain = prompt | llm



def extract_metadata(first_page_image: bytes) -> dict:
    import itertools
    import sys
    import threading
    import time

    def display_processing_message():
        for c in itertools.cycle(["|", "/", "-", "\\"]):
            if done:
                break
            sys.stdout.write("\rProcessing Image..." + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\rDone!\n")

    done = False
    t = threading.Thread(target=display_processing_message)
    t.start()
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": """The attached image is a first page of a research article.
                Extract the <title>, <the list of authors>, <the affiliation of the first author> and <abstract>. 
                <abstract> is the not summary of the paper or the abstract section, it is the whole OCRed characters contained in the abstract section without any modification or summarizing.
                Output should be a STRICTLY JSON object with the keys "title", "authors", "affiliation", and "abstract" as the following examle:
                
                Example of the output:
                
                {"title": "The use of restraint in mental health settings",
                 "authors": ["John Doe", "Jane Smith"], 
                 "affiliation": "University of California, Los Angeles", 
                 "abstract": "In these days, pharmacology is developing rapidly..."}.
        
                Never include any other text except the JSON object
                """,
            "images": [first_page_image],
            }
        ],
    )

    answer = response.get("message", {}).get("content", "").strip()
    
    if "```json" in answer and "```" in answer:
        start_idx = answer.find("```json") + 7
        end_idx = answer.find("```", start_idx)
        answer = answer[start_idx:end_idx].strip()
        
    try:
        final_answer = json.loads(answer)
    except json.JSONDecodeError:
        try:
            final_answer = chain.invoke({"input": answer}).model_dump()
        except json.JSONDecodeError:
            print(f"Error parsing JSON:\n {answer}\n")
            final_answer = {"title": "", "authors": [], "affiliation": "", "abstract": ""}
        except ValidationError:
            print(f"Validation error:\n {answer}\n")
            final_answer = {"title": "", "authors": [], "affiliation": "", "abstract": ""}
            
    done = True
    t.join()
    # print(final_answer)
    return final_answer


def _convert_to_dict(text: str) -> dict:
    try:
        answer = chain.invoke({"input": text})
    except json.JSONDecodeError:
        print(f"Error parsing JSON:\n {text}\n")
        return {"title": "", "authors": [], "affiliation": "", "abstract": ""}
    return answer