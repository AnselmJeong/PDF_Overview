import ollama
import json

VISION_MODEL = "llama3.2-vision"


def extract_metadata(first_page_image: bytes) -> dict:
    import itertools
    import sys
    import threading
    import time

    def display_processing_message():
        for c in itertools.cycle(["|", "/", "-", "\\"]):
            if done:
                break
            sys.stdout.write("\rProcessing " + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\rDone!     ")

    done = False
    t = threading.Thread(target=display_processing_message)
    t.start()
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": """The image is a first page of an academic article.
        Extract the title, the list of authors, the affiliation of only the first author and the abstract from the image.
        Output should be a STRICTLYJSON object in this format - {"title": <Title>, "authors": [<Authors>], "affiliation": <Affiliation>, "abstract": <Abstract>}.
        Do not output anything else except the JSON object""",
                "images": [first_page_image],
            }
        ],
    )

    answer = response.get("message", {}).get("content", "").strip()
    done = True
    t.join()
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        print(f"Error parsing JSON:\n {answer}\n")
        return {"title": "", "authors": [], "affiliation": "", "abstract": ""}
