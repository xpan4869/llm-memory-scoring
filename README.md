# llm-memory-scoring

This repository corresponds to a paper of "LLM-based scoring of narrative memories reveals that emotional arousal enhances central information at the expense of peripheral information"

The repository is organized as follows:

## Data  
- **Filmfest/** — short-film dataset
  - `1_annotations/` — event annotations + summary *(input)*
  - `2_arousal/` — human arousal ratings *(collected)* + LLM arousal outputs *(generated)*
  - `3_transcripts/` — free-recall transcripts *(input)*
  - `4_details/` — central/peripheral element lists *(generated)*
    - `central_detail_list/`
    - `peripheral_detail_list/`
  - `5_memory-fidelity/` — 0/1/2 scoring tables *(generated)*
- **Sherlock/** — same structure as Filmfest

## Scripts

- `scripts/arousal/1_rate_arousal_gpt4o.py` 
  Rates **event-level arousal** from annotations with an LLM (deterministic, `temperature=0`).

* `scripts/memory/1_generate_central_details.py`
  Generates **central (plot-relevant) elements** per event from annotations + summary.

* `scripts/memory/2_generate_peripheral_details.py`
  Generates **peripheral (descriptive) elements** per event.

* `scripts/memory/3_score_details.py`
  Scores free recall against the element lists (**0/1/2** fidelity), writes per-detail and aggregated tables.

### Instruction of Use
1. **Clone the Repository**
    ```bash
    git clone https://github.com/username/llm-memory-scoring.git
    cd /path/to/llm-memory-scoring
    ```

2. **Create a Virtual Environment (Optional but Recommended)**
    ```
    python -m venv .venv
    ```
    Activate the virtual environment:
    * On MACOS/Linux: 
      ```
      source venv/bin/activate
      ```
    * On Windows: 
      ```
      venv\Scripts\activate
      ```
    Environmental settings: 

3. **Install Dependencies**  
    
    Install the required packages using `pip`
    ```
    pip install -r requirements.txt
    ```

4. **Provide your API Key (recommended via .env)**
    ```
    echo "OPENAI_API_KEY=sk-..." > .env
    ```

### Usage

```bash
    DATASET=Filmfest # or Sherlock

    # Arousal (LLM, temperature=0)
    python3 scripts/arousal/1_rate_arousal_gpt4o.py --dataset "$DATASET"

    # Element lists
    python3 scripts/memory/1_generate_central_details.py    --dataset "$DATASET" --mem-type central
    python3 scripts/memory/2_generate_peripheral_details.py --dataset "$DATASET" --mem-type peripheral

    # Memory-fidelity scoring
    python3 scripts/memory/3_score_details.py --dataset "$DATASET" --mem-type central
    python3 scripts/memory/3_score_details.py --dataset "$DATASET" --mem-type peripheral
```
> **Outputs:**  
> • Arousal → `data/<DATASET>/2_arousal/`  
> • Elements → `data/<DATASET>/4_details/<mem_type>_detail_list/`  
> • Fidelity → `data/<DATASET>/5_memory-fidelity/<mem_type>_detail_scores/`
