# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 28, 2025
# Description: The script helps to use llm to rate how aroused an event is.

import os
import openai
import pandas as pd
import getpass
import glob

# ------------------ Hardcoded parameters ------------------ #
OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")
openai.api_key = OPENAI_API_KEY

os.chdir("/Users/yolandapan/automated-memory-scoring/scripts/arousal")
_THISDIR = os.getcwd()
DATASET_NAME = "Filmfest" # "Filmfest" or "Sherlock"
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, 'annotations'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, 'arousal'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ------------------ Define functions ------------------ #
def rate_event_arousal(transcript):
    prompt = f"""
    Arousal refers to when you are feeling very mentally or physically alert,  activated, and/or energized.
    Read the following description of a scene and rate the arousal  level of the scene on a scale of 1 to 10,
    With 1 being low arousal and 10 being high arousal.
    Please give a numeric rating. Only give the rating; no need to provide explanations.

    Scene:
    {transcript}
    """.strip()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()

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