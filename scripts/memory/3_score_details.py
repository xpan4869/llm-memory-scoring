# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: August 24, 2025
# Description: The script helps to generate scores for different participants of different events, for memory of central and peripheral details.

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
MEM_TYPE = "central" # "central" or "peripheral"
RECALL_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, '3_transcripts'))
DETAIL_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, f'4_details/{MEM_TYPE}_detail_list'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/' + DATASET_NAME, f'5_memory-fidelity/{MEM_TYPE}_detail_scores'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ------------------ Define functions ------------------ #
def generate_graded_central_scores(participant_id, participant_recall, event_number, central_details):
    prompt = f"""
    You are an expert annotator evaluating whether a participant recalled **central details** from a movie event.

    ---

    ### Participant Recall
    Participant `{participant_id}` recalled the following for Event `{event_number}`:
    \"\"\"{participant_recall}\"\"\"

    ---

    ### Central Details for Event {event_number}
    These are plot-essential facts or events. Your task is to assess **how accurately** each central detail is reflected in the participant’s recall.

    Use the following scoring scale:

    - **2** = Present: Clearly conveyed in the participant’s recall.
    - **1** = Partially Present: Partially conveyed or ambiguous.
    - **0** = Absent: Not mentioned or implied.

    ---

    **Central Detail Table:**
    {central_details}

    ---
    ### Instructions:
    Return **only** a Markdown table with these columns: `participants_id`, `event_number`, `central_id`, `score`

    Format the output like this:

    | participants_id | event_number | central_id | score |
    |-----------------|--------------|---------|-------|
    | {participant_id} | {event_number} | C1 | ? |
    | {participant_id} | {event_number} | C2 | ? |
    | ...             | ...          | ...        | ... |
    """.strip()

    response = openai.chat.completions.create(
      model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content

def generate_graded_peripheral_scores(participant_id, participant_recall, event_number, peripheral_details):
    prompt = f"""
    You are an expert annotator evaluating whether a participant recalled **peripheral details** from a movie event.

    ---

    ### Participant Recall
    This is what Participant `{participant_id}` remembered for Event `{event_number}`:
    \"\"\"{participant_recall}\"\"\"

    ---


    ### Peripheral Details for Event {event_number}
    These are **minor, descriptive features** of the event. They do **not change the plot**, but add context or sensory richness. Examples may include expressions, minor gestures, positioning, or manner of action.

    Use the following scoring scale:

    - **2 = Present**: The detail is clearly described in the participant's recall.
    - **1 = Partially Present**: The detail is vaguely or partially mentioned.
    - **0 = Absent**: The detail is not mentioned or implied at all.

    Detail Table:
    {peripheral_details}

    ### Instructions:
    Return **only** a Markdown table with these columns: `participants_id`, `event_number`, `peripheral_id`, `score`

    Format the table like this:

    | participants_id | event_number | peripheral_id | score |
    |-----------------|--------------|-----------|-------|
    | {participant_id} | {event_number} | P1 | ?
    | {participant_id} | {event_number} | P2 | ?
    | ...      | ...             | ...          | ...       | ...   |
    """.strip()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return response.choices[0].message.content

def parse_central_score_table(gpt_output: str):
    lines = gpt_output.strip().splitlines()
    scores = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Detect start of table
        if line.startswith("| participants_id"):
            in_table = True
            continue  # skip header row

        # Skip the separator row (e.g., |----|----|)
        if in_table and line.startswith("|--"):
            continue

        # Parse table rows
        if in_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 4:
                participant_id, event_number, central_id, score = parts
                scores.append({
                    "participant_id": participant_id,
                    "event_number": event_number,
                    "central_id": central_id,
                    "score": int(score) if score.isdigit() else score
                })

    return scores

def parse_peripheral_score_table(gpt_output: str):
    lines = gpt_output.strip().splitlines()
    scores = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Detect start of table
        if line.startswith("| participants_id"):
            in_table = True
            continue  # skip header row

        # Skip the separator row (e.g., |----|----|)
        if in_table and line.startswith("|--"):
            continue

        # Parse table rows
        if in_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 4:
                participant_id, event_number, peripheral_id, score = parts
                scores.append({
                    "participant_id": participant_id,
                    "event_number": event_number,
                    "peripheral_id": peripheral_id,
                    "score": int(score) if score.isdigit() else score
                })

    return scores

def read_recall_file(file_path):
    df = pd.read_csv(file_path)
    participant_id = os.path.basename(file_path).split('_recall_')[0]
    return df, participant_id

def parse_recall(df):
  transcript_by_event = []
  for index, row in df.iterrows():
    event_number = row.get("events")
    transcript = row.get("transcript")
    transcript_by_event.append((event_number, transcript))
  return transcript_by_event

def parse_table_by_event(table, event_number):
  parsed_table = table[table["event_number"] == event_number]
  return parsed_table


# ------------------- Main ------------------ #
if __name__ == "__main__":
    detail_files = glob.glob(os.path.join(DETAIL_PATH, '*.csv'))
    detail_df = pd.read_csv(detail_files[0])
    recall_files = glob.glob(os.path.join(RECALL_PATH, '*.csv'))

    id_col = 'central_id' if MEM_TYPE == 'central' else 'peripheral_id'

    # iterate over participants （files）
    all_results = []
    for recall_path in recall_files:
        df, participant_id = read_recall_file(recall_path)
        transcript_by_event = parse_recall(df)
        event_ids = [ev[0] for ev in transcript_by_event]
        number_events = len(set(event_ids))
        print(f"{participant_id}: {number_events} events")

        # iterate over events
        results = []
        for i in range(number_events):
            event_number = transcript_by_event[i][0]
            participant_recall = transcript_by_event[i][1]

            detail_table = parse_table_by_event(detail_df, event_number)

            # Skip events without recalls and record them as 0
            if pd.isna(participant_recall):
                if not detail_table.empty:
                    for did in detail_table[id_col].astype(str).tolist():
                        results.append({
                            "participant_id": participant_id,
                            "event_number": event_number,
                            id_col: did,
                            "score": 0
                        })
                continue
            
            if MEM_TYPE == 'central':
                gpt_output = generate_graded_central_scores(participant_id, participant_recall, event_number, detail_table)
                output = parse_central_score_table(gpt_output)
            else:
                gpt_output = generate_graded_peripheral_scores(participant_id, participant_recall, event_number, detail_table)
                output = parse_peripheral_score_table(gpt_output)
            for row in output:
                row["participant_id"] = participant_id
                row["event_number"] = event_number
            results.extend(output)
        
        results_df = pd.DataFrame(results)
        print(f"{participant_id} done")
        all_results.append(results_df)
    
    all_combined = pd.concat(all_results, ignore_index=True)
    all_combined = all_combined.sort_values(by=["participant_id", "event_number"])
    all_combined.to_csv(f"{SAVE_PATH}/graded_{MEM_TYPE}_scores_compiled.csv", index=False)
    print("All participant scores saved to one CSV.")
