# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 30, 2025
# Description: The script helps to generate a list of gists for different events from event annotations.

import os
import openai
import pandas as pd
import getpass
import glob

# ------------------ Hardcoded parameters ------------------ #
OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")
openai.api_key = OPENAI_API_KEY

os.chdir("/Users/yolandapan/automated-memory-scoring/scripts/memory")
_THISDIR = os.getcwd()
DATASET_NAME = "Filmfest" # "Filmfest" or "Sherlock"
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, '1_annotations'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, '4_details/central_detail_list'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ------------------ Define functions ------------------ #
def generate_central_details(annotation):
  prompt = f'''
  **Task:**
  You are assisting a memory researcher in analyzing a movie scene annotation to extract its **central details**.

  ---
  **Definition of Central Details:**

  * A **central detail** is an event or fact that is **causally or narratively essential** to understanding the scene.
  * These details are **plot-driving**: if removed, the meaning, sequence, or logic of the story would break down.
  * Central details can be **concrete or abstract**, but they must convey **key events**, **motivations**, or **outcomes**.
  * Avoid including **peripheral details** — sensory descriptions, background setting, or minor actions that do not impact the core story logic.
  ---

  **Reference Examples:**

  **Example A: Crashing the Bicycle**
  **Annotation:**
  A boy and his dad are riding a bike, with the boy sitting on the handlebars. They are going down a hill. The father squeezes the brakes to slow them down. He realizes that the brakes are broken. They both scream as the bike accelerates. The dad tries to brake with his shoes, without success. They hit a tree on the side of the road at full speed and fall off the bike.

  **Central Details (5):**

  * Man and boy on bicycle going down a hill
  * Man finds out brakes don’t work
  * Man tries to slow down
  * They crash into tree
  * They fall off of bike

  **Example B: Woman Squeezing Food**
  **Annotation:**
  An elderly woman is in a food store, handling a peach. She squeezes the fruit so hard that it bursts and splatters her in the face. The man behind the counter gives her an angry look. Embarrassed, she vanishes down one of the aisles, while he follows her. She starts squeezing a soft cheese with her thumbs, looking delighted.

  **Central Details (7):**

  * Woman in grocery store
  * Woman is squeezing a peach
  * Woman squeezes so hard juice squirts out
  * Cashier is angry/surprised
  * Woman runs away
  * Cashier follows her
  * Woman squeezes cheese

  ---

  **Annotation to be Analyzed:**
  \"\"\" {annotation}\"\"\"

  ---

  **Steps:**
  1. Identify distinct events or facts that are **essential for understanding what happens** in the scene.
  2. Avoid including background descriptions, visual modifiers, or minor emotional/sensory additions.
  3. Keep each idea **simple**, **non-redundant**, and focused on narrative flow.

  ---

  **Deliverable:**
  Provide a table with gist ideas, formatted like this:

  | Central ID | Idea Unit |
  |-----------|----------------|
  | C1        | ...            |
  | C2        | ...            |
  '''.strip()

  response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[
            {"role": "user", "content": prompt}
        ],
      temperature=0.0)

  return response.choices[0].message.content

def parse_central_detail_table(gpt_output: str, event_number=None):
    lines = gpt_output.strip().splitlines()
    central = []
    for line in lines:
        if line.strip().startswith("| C") and "|" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 2:
                central_id, idea_unit = parts
                central.append({
                    "event_number": event_number,
                    "central_id": central_id,
                    "central_content": idea_unit
                })
    return central

def flatten_central_data(raw_data):
    flat_list = []

    for sublist in raw_data:
        for row in sublist:
            if row["central_id"] != "Central ID":  # Skip the header row
                flat_list.append({
                    "event_number": row["event_number"],
                    "central_id": row["central_id"],
                    "central_content": row["central_content"]
                })

    return pd.DataFrame(flat_list)

# ------------------- Main ------------------ #
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DAT_PATH, '*.csv'))
    if csv_files:
        annotation_file = csv_files[0]
        annotation = pd.read_csv(annotation_file)
        print("Loaded:", annotation_file)
    else:
        print("No CSV file found.")
        exit()

    central_tables_all = []
    for idx, row in annotation.iterrows():
        event_number = row['event_number']
        annotation_text = row['annotation']
        central_table = parse_central_detail_table(generate_central_details(annotation_text), event_number)
        central_tables_all.append(central_table)

    central_df = flatten_central_data(central_tables_all)
    central_df.to_csv(f"{SAVE_PATH}/{DATASET_NAME}_balanced_central_detail_table.csv", index=False)
