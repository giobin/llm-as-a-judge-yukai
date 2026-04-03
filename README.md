# LLM as a Judge Pipeline

Pipeline Python per valutare un modello usato come **LLM-as-a-Judge**.

Funzionalita' principali:
- CLI con `argparse`
- backend modello `openai`, `vllm` o `vllm_in_process`
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
- `src/llm_judge_pipeline/model_client.py`: client OpenAI/vLLM/vLLM in-process
- `src/model_inference/cli.py`: entrypoint CLI multimodale per generazione risposte
- `src/model_inference/model_client.py`: client multimodale OpenAI/vLLM/vLLM in-process
- `src/common/vllm_in_process.py`: helper per inizializzare `vllm.LLM` in-process
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

## Esempio uso con vLLM server

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

## Esempio uso con vLLM in-process

```bash
CUDA_VISIBLE_DEVICES=0,1 llm-judge \
  --backend vllm_in_process \
  --model-name google/gemma-3-27b-it \
  --dataset-name caput/MAIA_dev_set_eng \
  --dataset-subset gen \
  --split train \
  --max-samples 20 \
  --prompt-file prompts/image_prompts/judge_prompt_en.txt \
  --vllm-tensor-parallel-size 2 \
  --vllm-enforce-eager \
  --vllm-disable-custom-all-reduce \
  --output-file judge_eval_report.json
```

Flag specifici per `vllm_in_process` e riusabili anche in `model-infer`:
- `--vllm-tensor-parallel-size`
- `--vllm-gpu-memory-utilization`
- `--vllm-max-model-len`
- `--vllm-trust-remote-code`
- `--vllm-enforce-eager`
- `--vllm-disable-custom-all-reduce`

## Model Inference

La CLI `model-infer` genera risposte a partire da:

Per i dataset image-only `VillanovaAI/multi-pixmo-cap` e `VillanovaAI/multi-pixmo-ask-model-anything` e' disponibile anche un flusso offline:

```bash
prefetch-multi-pixmo --max-images 400
```

Questo comando scarica in `HF_HOME` i due dataset Hugging Face e salva le prime 400 immagini di ogni subset (`it`, `en`, `es`) sotto `HF_HOME/llm_as_a_judge_assets/images/...`.
Poi le pipeline possono essere lanciate sui nodi GPU senza accesso internet usando:

```bash
model-infer \
  ... \
  --hf-local-files-only \
  --require-local-images
```

Se vuoi usare una root diversa per le immagini locali, aggiungi `--image-cache-root /path/dedicato`.
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

Esempio vLLM server:

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

Esempio vLLM in-process:

```bash
CUDA_VISIBLE_DEVICES=0,1 model-infer \
  --backend vllm_in_process \
  --model-name google/gemma-3-27b-it \
  --dataset-name caput/MAIA_ita \
  --dataset-subset gen \
  --split test \
  --videos-dir /data01/gbonetta/MAIA-Multimodal_AI_Assessment/Videos \
  --prompt-file prompts/video_prompts/generation_prompt_ita.txt \
  --num-frames 8 \
  --vllm-tensor-parallel-size 2 \
  --vllm-enforce-eager \
  --vllm-disable-custom-all-reduce \
  --output-file model_inference_ita.json
```

## Smoke Test GPU

Per fare uno smoke test di `model-infer` sui dataset `VillanovaAI/multi-pixmo-cap` e `VillanovaAI/multi-pixmo-ask-model-anything` su nodo GPU, con 10 sample per dataset e un modello alla volta con backend `vllm_in_process`, e' disponibile il launcher:

```bash
export LLM_JUDGE_SIF=/leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif
export HF_HOME=/leonardo_work/FBKLM_prj1/hf_cache
./scripts/run_multi_pixmo_infer_smoke_on_gpu.sh
```

Default principali:
- `NUM_GPUS=4`
- `MAX_SAMPLES=10`
- `SUBSETS=it`
- tutti i modelli elencati nello script worker

Output principali:
- report JSON di inferenza in `model_infer_results/smoke/`
- log per modello/dataset in `logs/multi_pixmo_smoke/`
- riepilogo finale in `logs/multi_pixmo_smoke/summary.tsv`

Per cambiare subset o limitare i modelli:

```bash
SUBSETS=it,en MAX_SAMPLES=10 MODELS=models--google--gemma-3-12b-it,models--Qwen--Qwen2.5-VL-7B-Instruct ./scripts/run_multi_pixmo_infer_smoke_on_gpu.sh
```

Il riepilogo include anche il tempo effettivo per 10 sample e una stima lineare per 240 sample.
Per il backend `vllm_in_process`, il launcher usa il Python nativo del container e aggiunge `src/` e i site-packages della `.venv` del repo al `sys.path`, evitando conflitti tra `vllm/transformers` del container e le librerie Python del progetto.

## Judge su risposte generate

`llm-judge` supporta input JSON prodotto da `model-infer`.

Flag nuovi:
- `--input-json`: path al JSON locale
- `--candidate-source`: `synthetic` (default), `generated` oppure `pixmo_transcripts`
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
  --prompt-file prompts/image_prompts/judge_prompt_en.txt \
  --output-file judge_eval_generated_eng.json
```

In modalita' `generated`, la candidate answer e' presa da `generated_answer1` e l'`expected_label` e' `yes`.

Per l'esperimento `multi-pixmo-cap` basato su transcript interni al dataset e' disponibile anche un source dedicato:
- usa solo sample con almeno 2 elementi in `transcripts`
- usa `transcripts[1]` come unica reference
- usa `transcripts[0]` come candidate positiva
- usa un transcript casuale da un altro sample come candidate negativa
- la quota di positivi e' controllata da `--pixmo-transcript-positive-ratio`

Per `VillanovaAI/multi-pixmo-cap` puoi anche usare la modalita' `transcript_pair`: se `transcripts` contiene almeno 2 elementi, il primo viene usato come `candidate_answer` e il secondo come unica reference. In questa modalita' `--generated-field` viene ignorato.

Esempio `generated + transcript_pair`:

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

Esempio `pixmo_transcripts`:

```bash
llm-judge \
  --backend openai \
  --model-name gpt-5.2 \
  --dataset-name VillanovaAI/multi-pixmo-cap \
  --dataset-subset it \
  --split train \
  --candidate-source pixmo_transcripts \
  --pixmo-transcript-positive-ratio 0.4 \
  --random-seed 42 \
  --prompt-file prompts/image_prompts/judge_prompt_ita.txt \
  --output-file judge_eval_pixmo_transcripts_it.json
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
