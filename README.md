# LLM as a Judge Pipeline

Pipeline Python per valutare un modello usato come **LLM-as-a-Judge**.

Funzionalita' principali:
- CLI con `argparse`
- backend modello `openai` o `vllm`
- caricamento dataset Hugging Face con o senza subset/config
- prompt esterno caricabile da file
- pipeline multimodale `model-infer` (video -> frame -> prompt -> risposta)
- generazione sample positivi/negativi per misurare la qualita' del judge
- supporto judge su `generated_answer1` da JSON locale (`--candidate-source generated`)
- metriche: accuracy, precision, recall, F1
- progress bar durante la valutazione
- modalita' verbose per debug prompt/risposte

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
- `prompts/image_prompts/`: prompt per task immagine / judge
- `prompts/video_prompts/`: prompt per task video / generation

## Installazione locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Nota: per la pipeline multimodale e' richiesta la libreria `decord`.

## Container su Leonardo

Il workflow container funzionante e' documentato in [container/README.md](/leonardo_scratch/fast/FBKLM_prj1/PROJECTS/llm-as-a-judge-yukai/container/README.md).
In breve:
- build del `.sif` sul login node
- bootstrap della `.venv` del progetto sul login node con `./container/run_in_container.sh --setup`
- download dei modelli Hugging Face nella cache condivisa `HF_HOME`
- avvio di `vllm serve` sul nodo GPU con `./container/run_vllm_server_in_container.sh ...`
- uso di `llm-judge` e `model-infer` contro `http://127.0.0.1:8000`

Variabili consigliate:

```bash
export LLM_JUDGE_SIF=/leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif
export HF_HOME=/leonardo_work/FBKLM_prj1/hf_cache
```

## Esempio uso con OpenAI

```bash
export OPENAI_API_KEY="..."
llm-judge \
  --backend openai \
  --model-name gpt-5-mini-2025-08-07 \
  --dataset-name giobin/MAIA_dev_set_eng_2wrong \
  --split train \
  --max-samples 20 \
  --prompt-file prompts/image_prompts/judge_prompt_en.txt \
  --verbose \
  --output-file judge_eval_report.json
```

Per dataset con subset/config, aggiungi `--dataset-subset gen`.

## Esempio uso con vLLM

```bash
llm-judge \
  --backend vllm \
  --model-name google/gemma-3-12b-it \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key EMPTY \
  --dataset-name giobin/MAIA_dev_set_eng_2wrong \
  --split train \
  --prompt-file prompts/image_prompts/judge_prompt_en.txt \
  --output-file judge_eval_report.json
```

## Model Inference

La CLI `model-infer` genera risposte a partire da:
- domanda (`question`)
- frame estratti dal video associato (`video_id` / `file_name`)

Regole principali:
- `--videos-dir` obbligatorio per dataset video
- sampling uniforme di `--num-frames` frame
- prompt da file con placeholder `{question}`
- token `<video>` opzionale nel prompt

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
  --prompt-file prompts/video_prompts/generation_prompt_en.txt \
  --num-frames 8 \
  --output-file model_inference_eng.json
```

Esempio vLLM:

```bash
model-infer \
  --backend vllm \
  --model-name google/gemma-3-12b-it \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key EMPTY \
  --dataset-name caput/MAIA_ita \
  --dataset-subset gen \
  --split test \
  --videos-dir /data01/gbonetta/MAIA-Multimodal_AI_Assessment/Videos \
  --prompt-file prompts/video_prompts/generation_prompt_ita.txt \
  --num-frames 8 \
  --output-file model_inference_ita.json
```

## Judge su risposte generate

`llm-judge` supporta input JSON prodotto da `model-infer`.

Esempio:

```bash
llm-judge \
  --backend openai \
  --model-name gpt-5.2 \
  --candidate-source generated \
  --input-json model_inference_eng.json \
  --generated-field generated_answer1 \
  --num-references 4 \
  --prompt-file prompts/image_prompts/judge_prompt_en.txt \
  --output-file judge_eval_generated_eng.json
```

## Dataset augmentation

```bash
export OPENAI_API_KEY="..."
export HF_TOKEN="..."
dataset-augment-foils \
  --source-dataset caput/MAIA_dev_set_eng \
  --target-owner giobin \
  --subset gen \
  --openai-model gpt-5.2
```

## Prompt optimization

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
