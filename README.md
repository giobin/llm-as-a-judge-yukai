# LLM as a Judge Pipeline

Pipeline Python per valutare un modello usato come **LLM-as-a-Judge**.

Funzionalità principali:
- CLI con `argparse`
- backend modello `openai` o `vllm`
- caricamento dataset Hugging Face con o senza subset/config
- prompt esterno caricabile da file
- pipeline multimodale `model-infer` (video -> frame -> prompt -> risposta)
- generazione sample positivi/negativi per misurare la qualità del judge
- supporto judge su `generated_answer1` da JSON locale (`--candidate-source generated`)
- metriche: accuracy, precision, recall, F1
- progress bar durante la valutazione
- modalità verbose per debug prompt/risposte

## Struttura

- `src/llm_judge_pipeline/cli.py`: entrypoint CLI
- `src/llm_judge_pipeline/model_client.py`: client OpenAI/vLLM
- `src/llm_judge_pipeline/dataset.py`: estrazione sample dal dataset
- `src/llm_judge_pipeline/prompting.py`: rendering prompt + parser output judge
- `src/llm_judge_pipeline/evaluation.py`: loop di valutazione + metriche
- `src/model_inference/cli.py`: entrypoint CLI multimodale per generazione risposte
- `src/model_inference/video.py`: risoluzione video locale + sampling frame uniforme
- `src/model_inference/prompting.py`: rendering prompt + integrazione `<video>`
- `src/model_inference/model_client.py`: client multimodale OpenAI/vLLM
- `prompts/judge_prompt.txt`: prompt template modificabile
- `prompts/judge_prompt_en.txt`: template inglese
- `prompts/judge_prompt_es.txt`: template spagnolo
- `prompts/generation_prompt_ita.txt`: template generazione ITA
- `prompts/generation_prompt_en.txt`: template generazione ENG
- `prompts/generation_prompt_es.txt`: template generazione ESP

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Nota: per la pipeline multimodale e' richiesta la libreria `decord` (installata dalle dipendenze del progetto).

## Esempio uso con OpenAI

```bash
export OPENAI_API_KEY="..."
llm-judge \
  --backend openai \
  --model-name gpt-5-mini-2025-08-07 \
  --dataset-name giobin/MAIA_dev_set_eng_2wrong \
  --split train \
  --max-samples 20 \
  --prompt-file prompts/judge_prompt_en.txt \
  --verbose \
  --output-file judge_eval_report.json
```

Per dataset con subset/config (es. MAIA originale), aggiungi `--dataset-subset gen`.
Per saltare le prime N righe del dataset e valutare dalla riga successiva, usa `--offset-samples N` (poi `--max-samples` si applica da quell'offset).

## Esempio uso con vLLM (server OpenAI-compatible)

```bash
llm-judge \
  --backend vllm \
  --model-name openai/gpt-oss-20b #meta-llama/Llama-3.1-8B-Instruct \
  --vllm-base-url http://localhost:8000 \
  --dataset-name caput/MAIA_dev_set_eng \
  --dataset-subset gen \
  --split train \
  --max-samples 20
```

## Esempio uso con vLLM in-process

```bash
CUDA_VISIBLE_DEVICES=0,1 llm-judge \
  --backend vllm_in_process \
  --model-name google/gemma-3-27b-it \
  --vllm-tensor-parallel-size 2 \
  --vllm-enforce-eager \
  --vllm-disable-custom-all-reduce \
  --dataset-name caput/MAIA_dev_set_eng \
  --dataset-subset gen \
  --split train \
  --max-samples 20
```

## Model Inference (Multimodale MAIA GEN)

La CLI `model-infer` genera risposte a partire da:
- domanda (`question`)
- frame estratti dal video associato (`video_id` / `file_name`)

Regole principali:
- scope v1: solo task `gen`
- `--videos-dir` obbligatorio (video locali `VideoX.mp4`)
- sampling uniforme di `--num-frames` frame (default `8`)
- prompt da file con placeholder `{question}`
- token `<video>` opzionale:
  - se presente, i frame vengono inseriti nel punto del token
  - se assente, i frame vengono prependati al messaggio user

Esempio OpenAI:

```bash
export OPENAI_API_KEY="..."
model-infer \
  --backend openai \
  --model-name gpt-4o-mini \
  --dataset-name caput/MAIA_eng \
  --dataset-subset gen \
  --split test \
  --videos-dir /data01/gbonetta/maia_cache/Videos \
  --prompt-file prompts/generation_prompt_en.txt \
  --num-frames 8 \
  --output-file model_inference_eng.json
```

Esempio vLLM OpenAI-compatible:

```bash
model-infer \
  --backend vllm \
  --model-name OpenGVLab/InternVL3-8B \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key EMPTY \
  --dataset-name caput/MAIA_ita \
  --dataset-subset gen \
  --split test \
  --videos-dir /data01/gbonetta/MAIA-Multimodal_AI_Assessment/Videos \
  --prompt-file prompts/generation_prompt_ita.txt \
  --num-frames 8 \
  --output-file model_inference_ita.json
```

Esempio vLLM in-process:

```bash
CUDA_VISIBLE_DEVICES=0,1 model-infer \
  --backend vllm_in_process \
  --model-name google/gemma-3-27b-it \
  --dataset-name caput/MAIA_ita \
  --dataset-subset gen \
  --split test \
  --videos-dir /data01/gbonetta/MAIA-Multimodal_AI_Assessment/Videos \
  --prompt-file prompts/generation_prompt_ita.txt \
  --num-frames 8 \
  --vllm-tensor-parallel-size 2 \
  --vllm-enforce-eager \
  --vllm-disable-custom-all-reduce \
  --output-file model_inference_ita.json
```

Output `model-infer`:
- JSON object con:
  - `schema_version`
  - `config`
  - `summary`
  - `rows`
- ogni riga contiene i campi originali +:
  - `generated_answer1`
  - `generation_error`
  - `generation_meta` (`video_id`, `video_path`, `num_frames_used`, `frame_indices`)

## Judge Su Risposte Generate

`llm-judge` supporta input JSON prodotto da `model-infer`.

Flag nuovi:
- `--input-json`: path al JSON locale
- `--candidate-source`: `synthetic` (default) oppure `generated`
- `--generated-field`: campo da usare come candidate (default `generated_answer1`)
- `--generated-sample-mode`: `generated_field` (default) oppure `transcript_pair`

Esempio:

```bash
llm-judge \
  --backend openai \
  --model-name gpt-5.2 \
  --candidate-source generated \
  --input-json model_inference_eng.json \
  --generated-field generated_answer1 \
  --generated-sample-mode generated_field \
  --num-references 4 \
  --prompt-file prompts/judge_prompt_en.txt \
  --output-file judge_eval_generated_eng.json
```

In modalita' `generated`, la candidate answer e' presa da `generated_answer1` e l'`expected_label` e' `yes`.

Per `VillanovaAI/multi-pixmo-cap` puoi anche usare la modalita' `transcript_pair`: se `transcripts` contiene almeno 2 elementi, il primo viene usato come `candidate_answer` e il secondo come unica reference. In questa modalita' `--generated-field` viene ignorato.

Esempio:

```bash
llm-judge \
  --backend openai \
  --model-name gpt-5.2 \
  --candidate-source generated \
  --input-json model_inference_multi_pixmo_cap.json \
  --generated-sample-mode transcript_pair \
  --prompt-file prompts/image_prompts/qa_judge_prompt_en.txt \
  --output-file judge_eval_multi_pixmo_cap_transcripts.json
```

## Note sul dataset

La pipeline assume che ogni riga abbia:
- `question`
- una o più colonne `answer*` (es. `answer1`, `answer2`, ...)

Le prime 4 risposte valide vengono usate come ground truth.
La candidate answer positiva viene scelta tra le risposte valide rimanenti (non presenti nelle 4 ground truth).

Per testare il judge vengono creati:
- sample **positivi**: `candidate_answer` presa da risposte corrette non incluse nelle 4 reference (label attesa `yes`)
- sample **negativi**:
  - se presenti, usa `wrong_answer1` / `wrong_answer2` (label attesa `no`)
  - altrimenti fallback a una risposta presa da un'altra domanda

## Output

Viene scritto un JSON (`judge_eval_report.json` di default) con:
- configurazione usata
- metriche aggregate
- predizioni per ogni esempio

Con `--verbose`, per ogni sample vengono stampati:
- prompt completo inviato al judge
- risposta raw del modello
- parse minimale (`verdict`, `score`) e label attesa

## Dataset Augmentation (wrong answers)

E' disponibile una CLI separata per creare un dataset con due colonne aggiuntive:
- `wrong_answer1`: foil minimamente sbagliato a partire da `answer1`
- `wrong_answer2`: foil minimamente sbagliato a partire da `answer2`

La CLI usa OpenAI (default model `gpt-5.2`) e pubblica su Hugging Face Hub un dataset con nome:
- `<target_owner>/<original_dataset_name>_2wrong`
- esempio: `caput/MAIA_dev_set_eng` -> `giobin/MAIA_dev_set_eng_2wrong`

Comando:

```bash
export OPENAI_API_KEY="..."
export HF_TOKEN="..."
dataset-augment-foils \
  --source-dataset caput/MAIA_dev_set_eng \
  --target-owner giobin \
  --subset gen \
  --openai-model gpt-5.2
```

Opzioni utili:
- `--max-rows N`: limita righe per split (debug)
- `--no-push`: genera ma non pubblica su Hub
- `--verbose`: stampa question/answers/foils per ogni riga

## Prompt Optimization (PromptWizard)

E' disponibile un sottoprogetto in `src/prompt_optimization` per ottimizzare il prompt del judge con PromptWizard, seguendo lo scenario con training data + in-context examples.

Workflow implementato:
- carica `giobin/MAIA_dev_set_ita_2wrong` (o altro dataset compatibile)
- crea esempi binari `yes/no` per il task judge
- split 50/50 train-test
- lancia PromptWizard (`get_best_prompt` con `use_examples=True`)
- valuta su test (`evaluate`)
- salva:
  - `best_prompt_raw.txt` (output grezzo PromptWizard)
  - `best_prompt.txt` (prompt finale pronto per `llm-judge`, con placeholder)
  - `expert_profile.txt`
  - summary in JSON

Il prompt finale (`best_prompt.txt`) e' normalizzato per la pipeline `llm-judge`:
- output richiesto in JSON (`verdict: yes|no`)
- contiene sempre i placeholder:
  - `Question: {question}`
  - `Candidate answer: {candidate_answer}`
  - `Ground truth answers: {ground_truth_answers}`

Prerequisiti:
- PromptWizard disponibile in locale (default path: `/data01/gbonetta/PromptWizard`)
- variabile `OPENAI_API_KEY` impostata

Comando:

```bash
export OPENAI_API_KEY="..."
optimize-judge-prompt \
  --scenario 3 \
  --dataset-name giobin/MAIA_dev_set_ita_2wrong \
  --split train \
  --train-ratio 0.5 \
  --openai-model gpt-5-mini-2025-08-07 \
  --promptwizard-root /data01/gbonetta/PromptWizard \
  --work-dir src/prompt_optimization/artifacts/maia_ita \
  --verbose
```

Scenari supportati:
- `--scenario 1`: no training data, no in-context examples (PromptWizard stampa variazioni candidate; test accuracy non disponibile).
- `--scenario 2`: no training data iniziale, genera esempi sintetici e poi ottimizza con in-context examples (valutazione su test set reale).
- `--scenario 3`: training data reale + in-context examples (valutazione su test set reale).
