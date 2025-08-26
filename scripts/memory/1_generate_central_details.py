# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: August 26, 2025
# Description: The script helps to generate a list of gists for different events from event annotations.

import os, sys, argparse
import openai
import pandas as pd
from pathlib import Path

# ---------- env & API key ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("[ERROR] OPENAI_API_KEY not found. Put it in .env or export it before running.")
openai.api_key = api_key

# ---- dataset & paths ----
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--dataset", choices=["Filmfest", "Sherlock"])
_args, _ = _ap.parse_known_args()
DATASET_NAME = _args.dataset or os.getenv("DATASET", "Filmfest")

REPO = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
DS_ROOT = REPO / "data" / DATASET_NAME

DAT_PATH = DS_ROOT / "1_annotations"
if not DAT_PATH.exists():
    sys.exit(f"[ERROR] Not found: {DAT_PATH}")
SAVE_PATH = DS_ROOT / "4_details" / "central_detail_list"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ------------------ Define functions ------------------ #
PROMPT = '''
  **Task:**
  You are assisting a memory researcher in analyzing a movie scene annotation to extract its **central details**.

  ---
  **Definition of Central Details:**
    Central details are causally essential elements of a narrative that sustain the storyline. They include information that drives the plot forward, explains character motivations, or marks turning points in the story. Without these details, the coherence or progression of the narrative would be disrupted.
  ---

  **Central Storyline as Reference:**
  \"\"\" {summary}\"\"\"

  ---

  **Annotation to be Analyzed:**
  \"\"\" {annotation}\"\"\"

  ---

  **Steps:**
  1. Extract only details that are **causally essential** (plot-relevant).
  2. Exclude descriptive or atmospheric elements that enrich context but do not alter the storyline.
  3. Express each idea in a brief (≤10 words) form that captures its plot-relevant role.
  4. Keep them **non-redundant** and focused on the **narrative skeleton**.


  ---

  **Deliverable:**
  Provide a table with gist ideas, formatted like this:

  | Central ID | Idea Unit |
  |-----------|----------------|
  | C1        | ...            |
  | C2        | ...            |
  '''.strip()

def generate_central_details(summary, annotation):
  prompt = PROMPT.format(summary=summary, annotation=annotation)
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
    annotation_file = list(DAT_PATH.glob("*_annotations.csv"))[0] 
    summary_file = list(DAT_PATH.glob("*_summary.csv"))[0] 
    annotations = pd.read_csv(annotation_file)
    summaries = pd.read_csv(summary_file)

    central_tables_all = []
    for idx, row in annotations.iterrows():
        event_number = row['event_number']
        annotation_text = row['annotation']

        if 'movie_title' in annotations.columns and pd.notna(row.get('movie_title')):
            movie_title = row['movie_title']
            # filter summaries for that movie_title
            match = summaries.loc[summaries["movie_title"] == movie_title, "summary"]
            if not match.empty:
                summary = match.iloc[0]
            else:
                continue
        else:
            summary = summaries['summary'].iloc[0]
        
        central_table = parse_central_detail_table(generate_central_details(summary, annotation_text), event_number)
        central_tables_all.append(central_table)

    central_df = flatten_central_data(central_tables_all)
    central_df.to_csv(f"{SAVE_PATH}/{DATASET_NAME}_balanced_central_detail_table.csv", index=False)
