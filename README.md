# Automated Reasoning Challenge

Evaluate language model answers on tricky reasoning questions.

---

## Setup

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file**  
   (copy `.env.example` and fill in your values)
   ```
   # For generating model answers (baseline/workflow)
   API_KEY=xxx
   LLM=xxx
   BASE_URL=xxx

   # For answer evaluation (grading)
   EVAL_API_KEY=xxx
   EVAL_LLM=xxx
   EVAL_BASE_URL=xxx
   ```

---

## Running Experiments

Run all experiments with:
```bash
python -B arcbench/main.py
```

- Model answers are generated using the model defined by `LLM`, `API_KEY`, etc.
- Automatic grading is performed using the model specified by `EVAL_LLM`, `EVAL_API_KEY`, etc.
- Results are saved in `results/` and include the model name in the filename.

To try a different LLM for generation, edit the `LLM` in your `.env` and run again.  
To change the grading rubric, change the `EVAL_LLM` variables.

---

## Notes

- Make sure your input data exists (`data/ArcBench.jsonl`).
- Logs and errors are saved in the `results` and `logs` folders.
- By default, both baseline and workflow experiments will be run.  
  If you only want to run one, edit the bottom of `arcbench/main.py`.

