# scripts/arousal/1_rate_arousal_gpt4o.py
# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: August 26, 2025
# Description: Rate event-level arousal (1–10) from annotations via LLM (temperature=0).

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
SAVE_PATH = DS_ROOT / "2_arousal"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ------------------ Define functions ------------------ #
PROMPT = """ Arousal refers to when you are feeling very mentally or physically alert,  activated, and/or energized.
Read the following description of a scene and rate the arousal  level of the scene on a scale of 1 to 10,
With 1 being low arousal and 10 being high arousal.
Please give a numeric rating. Only give the rating; no need to provide explanations.

Scene:
    {transcript}
""".strip()

def rate_event_arousal(transcript):
    prompt = PROMPT.format(transcript=transcript)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    return response.choices[0].message.content.strip()

# ------------------- Main ------------------ #
if __name__ == "__main__":
    csv_files = list(DAT_PATH.glob("*_annotations.csv")) 
    if csv_files:
        annotation_file = csv_files[0]
        annotation = pd.read_csv(annotation_file)
        print("Loaded:", annotation_file)
    else:
        print("No CSV file found.")
        exit()
    annotations = pd.read_csv(annotation_file)

    results = []
    for idx, row in annotations.iterrows():
        event_number = row.get('event_number', None)
        annotation_text = row.get('annotation', '')

        # Check: skip if event_number is missing or NaN
        if pd.isna(event_number):
            print(f"Skipping row {idx} due to missing event_number.")
            continue

        score = rate_event_arousal(annotation_text)
        results.append({
            'event_number': event_number,
            'arousal_score': score
        })

    results_df = pd.DataFrame(results)
    output_file = os.path.join(SAVE_PATH, f"{DATASET_NAME}_arousal_gpt4o.csv")
    results_df.to_csv(output_file, index=False)