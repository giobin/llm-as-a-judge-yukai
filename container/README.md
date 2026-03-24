# Container Workflow

Workflow consigliato e testato su Leonardo:
- pull del container ufficiale `vllm/vllm-openai:v0.17.0` in un file `.sif`
- bootstrap della `.venv/` del progetto sul login node con internet
- download dei modelli Hugging Face nella cache condivisa
- avvio di `vllm serve` sul nodo GPU in modalita' offline
- esecuzione delle CLI del progetto nello stesso container

## File usati

- `build_container.sh`: crea il `.sif` con `singularity pull`
- `run_in_container.sh`: bootstrap della `.venv/` e run delle CLI Python del repo
- `run_vllm_server_in_container.sh`: avvio del server vLLM sul nodo GPU con `HF_HOME` forzata e offline mode

## 1. Costruzione del container sul login node

```bash
cd /leonardo_scratch/fast/FBKLM_prj1/PROJECTS/llm-as-a-judge-yukai
sbatch container/build_container.sh /leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif
```

Di default lo script usa `vllm/vllm-openai:v0.17.0`.
Per cambiare tag:

```bash
VLLM_VERSION=<tag> sbatch container/build_container.sh /path/to/llm-judge-vllm.sif
```

## 2. Variabili ambiente condivise

Sul login node e poi sul nodo GPU esporta sempre:

```bash
export LLM_JUDGE_SIF=/leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif
export HF_HOME=/leonardo_work/FBKLM_prj1/hf_cache
mkdir -p "$HF_HOME"
```

`HF_HOME` deve puntare a una cache condivisa visibile sia dal login node sia dai nodi GPU.

## 3. Bootstrap della `.venv` sul login node

La prima installazione delle dipendenze del progetto va fatta sul login node, perche' usa internet:

```bash
cd /leonardo_scratch/fast/FBKLM_prj1/PROJECTS/llm-as-a-judge-yukai
./container/run_in_container.sh --setup
```

Questo crea `.venv/` nel repo con `uv` e `--system-site-packages`, riusando `torch` e `vllm` dal container.

## 4. Download dei modelli Hugging Face

I modelli da servire con vLLM devono essere gia' in cache dentro `HF_HOME` prima di andare sul nodo GPU.
Per esempio, sul login node:

```bash
export HF_HOME=/leonardo_work/FBKLM_prj1/hf_cache
huggingface-cli login
huggingface-cli download google/gemma-3-12b-it
```

Se il modello e' gated, devi anche averne accettato i termini sul sito Hugging Face.

## 5. Avvio di un nodo GPU con SLURM

```bash
srun -C xorg -N1 --gres=gpu:1 -p boost_usr_prod -t 01:00:00 --account=FBKLM_prj1 --pty bash
```

## 6. Avvio di vLLM sul nodo GPU

```bash
cd /leonardo_scratch/fast/FBKLM_prj1/PROJECTS/llm-as-a-judge-yukai
export LLM_JUDGE_SIF=/leonardo_work/FBKLM_prj1/$USER/containers/llm-judge-vllm.sif
export HF_HOME=/leonardo_work/FBKLM_prj1/hf_cache
./container/run_vllm_server_in_container.sh google/gemma-3-12b-it --dtype bfloat16
```

Lo script:
- usa `singularity exec --nv`
- binda `HF_HOME`
- forza `HF_HOME`, `TRANSFORMERS_CACHE` e `HF_DATASETS_CACHE` dentro il container
- forza `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`
- filtra le path CUDA host da `LD_LIBRARY_PATH`

Check rapidi sul nodo GPU:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## 7. Esecuzione delle CLI del progetto nel container

Esempio `llm-judge`:

```bash
./container/run_in_container.sh -m llm_judge_pipeline.cli \
  --backend vllm \
  --model-name google/gemma-3-12b-it \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key EMPTY \
  --dataset-name giobin/MAIA_dev_set_eng_2wrong \
  --split train \
  --prompt-file prompts/image_prompts/judge_prompt_en.txt \
  --output-file judge_eval_report.json
```

Esempio `model-infer`:

```bash
./container/run_in_container.sh -m model_inference.cli \
  --backend vllm \
  --model-name google/gemma-3-12b-it \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-api-key EMPTY \
  --dataset-name caput/MAIA_eng \
  --dataset-subset gen \
  --split test \
  --videos-dir /data01/gbonetta/MAIA-Multimodal_AI_Assessment/Videos \
  --prompt-file prompts/video_prompts/generation_prompt_en.txt \
  --output-file model_inference_report.json
```

## Note operative

- Se cambi dipendenze Python del progetto, elimina `.venv/` o `.venv/.installed` e rilancia `./container/run_in_container.sh --setup` sul login node.
- Se cambi codice Python del repo, non serve ricostruire il `.sif`: il repo e' bind-mounted.
- Se vLLM prova ad andare online da nodo GPU, il problema non e' il bind di `HF_HOME` ma la cache modello non pronta o incompleta.
