# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: August 26, 2025
# Description: The script helps to generate a list of details/peripherals for different events from event annotations, matching the number of gists per event.

import os, sys, argparse
import openai
import pandas as pd
from collections import defaultdict
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
NUM_PATH = DS_ROOT / "4_details" / "central_detail_list"
if not NUM_PATH.exists():
    sys.exit(f"[ERROR] Not found: {NUM_PATH}")
SAVE_PATH = DS_ROOT / "4_details" / "peripheral_detail_list"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ------------------ Define functions ------------------ #
PROMPT = '''
    **Task:**
    You are assisting a memory researcher in analyzing a movie scene annotation to extract its **peripheral details**.
    
    ---
    **Definition of Peripheral Details:**
    Peripheral details are descriptive elements that enrich the narrative context but are not essential to its causal structure. They provide texture, atmosphere, or background information (e.g., setting descriptions or incidental features), yet their absence would not alter the core storyline or change character motivations.
    ---

    **Central Storyline as Reference:**
    \"\"\" {summary}\"\"\"

    ---

    **Annotation to be Analyzed:**
    \"\"\" {annotation} \"\"\"

    ---

    **Steps:**
    1. Identify distinct details that are **descriptive but not causally essential**.
    2. Exclude any plot-driving events, states, or turning points (those belong in central).
    3. Express each idea in a brief (≤10 words) form that captures its plot-relevant role.
    4. Avoid redundancy or interpretation (no camera notes, no analysis).
    5. Extract **exactly {num_details} details** — no more, no less.

    ---

    **Deliverable:**
    Provide a table with exactly {num_details} peripheral details, formatted like this:

    | Peripheral ID | Detail |
    |---------------|--------|
    | P1            | ...    |
    | P2            | ...    |
    | ...           | ...    |
'''.strip()

def generate_peripheral_details(summary, annotation, num_details):
    prompt = PROMPT.format(summary=summary, annotation=annotation, num_details=num_details)
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
    central_files = list(NUM_PATH.glob("*.csv"))
    if not central_files:
        raise FileNotFoundError("No *.csv file found in NUM_PATH")
    central_file = central_files[0]
    
    central_table = pd.read_csv(central_file)

    annotation_file = list(DAT_PATH.glob("*_annotations.csv"))[0] 
    summary_file = list(DAT_PATH.glob("*_summary.csv"))[0] 
    annotations = pd.read_csv(annotation_file)
    summaries = pd.read_csv(summary_file)

    event_central_counts = defaultdict(int)
    for _, row in central_table.iterrows():
        event_number = row['event_number']
        event_central_counts[event_number] += 1
    
    peripheral_table_all = []
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

        num_details = event_central_counts.get(event_number, 6)
        peripheral_table = parse_peripheral_detail_table(generate_peripheral_details(summary, annotation_text, num_details), event_number)
        peripheral_table_all.append(peripheral_table)
    
    peripheral_df = flatten_peripheral_data(peripheral_table_all)
    peripheral_df.to_csv(f"{SAVE_PATH}/{DATASET_NAME}_balanced_peripheral_detail_table.csv", index=False)
