# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 30, 2025
# Description: The script helps to generate a list of details/peripherals for different events from event annotations, matching the number of gists per event.

import os
import openai
import pandas as pd
import getpass
import glob
from collections import defaultdict


# ------------------ Hardcoded parameters ------------------ #
OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")
openai.api_key = OPENAI_API_KEY 

os.chdir("/Users/yolandapan/automated-memory-scoring/scripts/memory")
_THISDIR = os.getcwd()
DATASET_NAME = "Filmfest" # "Filmfest" or "Sherlock"
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, '1_annotations'))
NUM_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, '4_details/central_detail_list'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, '4_details peripheral_detail_list'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ------------------ Define functions ------------------ #
def generate_peripheral_details(annotation, num_details):
    prompt = f'''
    **Task:**
    You are assisting a memory researcher in analyzing a movie scene annotation to extract its **peripheral details**.
    
    ---

    **Definition of Peripheral Details:**
    Peripheral details are **minor descriptive elements** that **do not affect the core storyline** but provide **enriching context**. These are small, literal features that make the scene more vivid without changing its outcome or logic.

    Peripheral details may include:
    - Character gestures, facial expressions, or emotional reactions
    - Specific objects, body positions, or brief actions that are not plot-driving
    - Spatial location or manner in which something is done (e.g., speed, position)

    Do **not** include:
    - Plot-driving events or important narrative shifts (those are central)
    - Interpretive or thematic summaries
    - Any references to **camera angles**, **scene framing**, **shots**, or **zooming**

    ---

    **Reference Examples:**

    **Example A: Crashing the Bicycle**
    **Annotation:**
    A boy and his dad are riding a bike, with the boy sitting on the handlebars. They are going down a hill. The father squeezes the brakes to slow them down. He realizes that the brakes are broken. They both scream as the bike accelerates. The dad tries to brake with his shoes, without success. They hit a tree on the side of the road at full speed and fall off the bike.

    **Peripheral Details (5):**
    * Boy sitting on the handlebars
    * Both scream
    * With his shoes
    * On the side of the road
    * At full speed

    ---

    **Example B: Woman Squeezing Food**
    **Annotation:**
    An elderly woman is in a food store, handling a peach. She squeezes the fruit so hard that it bursts and splatters her in the face. The man behind the counter gives her an angry look. Embarrassed, she vanishes down one of the aisles, while he follows her. She starts squeezing a soft cheese with her thumbs, looking delighted.

    **Peripheral Details (7):**
    * So hard
    * In the face
    * Behind the counter
    * Down one of the aisles
    * Soft (cheese)
    * With her thumbs
    * Looking delighted

    ---

    **Annotation to be Analyzed:**
    \"\"\" {annotation} \"\"\"

    ---

    **Instructions:**
    1. Read the annotation carefully.
    2. Identify **exactly {num_details} peripheral details** that reflect **descriptive, non-essential elements** of the scene.
    3. Each detail should be a **short, simple phrase** (a few words or one sentence maximum).
    4. Avoid summarizing major events or plot changes — do **not include central details**.
    5. **Do not generate more than {num_details} details.**

    ---

    **Deliverable:**
    Provide a table with exactly {num_details} peripheral details, formatted like this:

    | Peripheral ID | Detail |
    |---------------|--------|
    | P1            | ...    |
    | P2            | ...    |
    | ...           | ...    |
'''.strip()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content

def parse_peripheral_detail_table(gpt_output: str, event_number=None):
    lines = gpt_output.strip().splitlines()
    details = []
    for line in lines:
        if line.strip().startswith("| P") and "|" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 2:
                peripheral_id, idea_unit = parts
                details.append({
                    "event_number": event_number,
                    "peripheral_id": peripheral_id,
                    "peripheral": idea_unit
                })
    return details

def flatten_peripheral_data(raw_data):
    flat_list = []

    for sublist in raw_data:
        for row in sublist:
            if row["peripheral_id"] != "Peripheral ID":  # Skip the header row
                flat_list.append({
                    "event_number": row["event_number"],
                    "peripheral_id": row["peripheral_id"],
                    "peripheral": row["peripheral"]
                })

    return pd.DataFrame(flat_list)

# ------------------- Main ------------------ #
if __name__ == "__main__":
    central_file = glob.glob(os.path.join(NUM_PATH, '*.csv'))
    annotation_file = glob.glob(os.path.join(DAT_PATH, '*.csv'))
    if annotation_file and central_file:
        central_table = pd.read_csv(central_file[0])
        print("Loaded:", central_file)
        annotation = pd.read_csv(annotation_file[0])
        print("Loaded:", annotation_file)
    elif not central_file:
        print("No CSV file found.")
        exit()
    elif not annotation_file:
        print("NO annotation file found.")
        exit()

    event_central_counts = defaultdict(int)
    for _, row in central_table.iterrows():
        event_number = row['event_number']
        event_central_counts[event_number] += 1
    
    peripheral_table_all = []
    for idx, row in annotation.iterrows():
        event_number = row['event_number']
        annotation_text = row['annotation']
        num_details = event_central_counts.get(event_number, 6)
        peripheral_table = parse_peripheral_detail_table(generate_peripheral_details(annotation_text, num_details), event_number)
        peripheral_table_all.append(peripheral_table)
    
    peripheral_df = flatten_peripheral_data(peripheral_table_all)
    peripheral_df.to_csv(f"{SAVE_PATH}/{DATASET_NAME}_balanced_peripheral_detail_table.csv", index=False)

