from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

from .dataset_builder import build_judge_examples, save_jsonl, split_train_test

PIPELINE_OUTPUT_INSTRUCTION = (
    'Restituisci SOLO JSON valido nel formato: '
    '{"verdict":"yes|no","score":0.0,"reason":"breve motivazione"}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize LLM-as-a-Judge prompt on MAIA 2wrong datasets via PromptWizard (Scenarios 1/2/3)."
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "1: no training data, no in-context examples; "
            "2: no training data + synthetic in-context examples; "
            "3: training data + in-context examples."
        ),
    )
    parser.add_argument("--dataset-name", default="giobin/MAIA_dev_set_ita_2wrong")
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="If >0, limit number of source rows.")

    parser.add_argument("--promptwizard-root", default="/data01/gbonetta/PromptWizard")
    parser.add_argument("--work-dir", default="src/prompt_optimization/artifacts/maia_ita")

    parser.add_argument("--openai-model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--openai-api-key", default=None)

    parser.add_argument("--seen-set-size", type=int, default=30)
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument("--mutation-rounds", type=int, default=2)
    parser.add_argument("--mutate-refine-iterations", type=int, default=3)
    parser.add_argument("--refine-task-eg-iterations", type=int, default=2)
    parser.add_argument("--style-variation", type=int, default=3)
    parser.add_argument("--questions-batch-size", type=int, default=1)
    parser.add_argument("--min-correct-count", type=int, default=3)
    parser.add_argument("--max-eval-batches", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=1)

    parser.add_argument(
        "--task-description",
        default=(
            "Sei un LLM-as-a-Judge. Devi valutare se una candidate answer e' corretta "
            "rispetto a una domanda e a risposte di riferimento."
        ),
    )
    parser.add_argument(
        "--base-instruction",
        default=(
            "Analizza domanda, candidate answer e reference answers. "
            "Decidi se la candidate e' semanticamente corretta rispetto ad almeno una reference."
        ),
    )
    parser.add_argument(
        "--answer-format",
        default=(
            "Restituisci SOLO JSON valido nel formato "
            '{"verdict":"yes|no","score":0.0,"reason":"breve motivazione"}.'
        ),
    )
    parser.add_argument("--experiment-name", default="maia_ita_judge_promptopt")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def write_promptopt_config(path: Path, args: argparse.Namespace, seen_set_size: int, few_shot_count: int) -> None:
    content = f"""prompt_technique_name: \"critique_n_refine\"
unique_model_id: promptopt-model
mutate_refine_iterations: {args.mutate_refine_iterations}
mutation_rounds: {args.mutation_rounds}
refine_instruction: true
refine_task_eg_iterations: {args.refine_task_eg_iterations}
style_variation: {args.style_variation}
questions_batch_size: {args.questions_batch_size}
min_correct_count: {args.min_correct_count}
max_eval_batches: {args.max_eval_batches}
top_n: {args.top_n}
task_description: {json.dumps(args.task_description, ensure_ascii=False)}
base_instruction: {json.dumps(args.base_instruction, ensure_ascii=False)}
answer_format: {json.dumps(args.answer_format, ensure_ascii=False)}
seen_set_size: {seen_set_size}
few_shot_count: {few_shot_count}
num_train_examples: 20
generate_reasoning: false
generate_expert_identity: true
generate_intent_keywords: false
"""
    path.write_text(content, encoding="utf-8")


def extract_optimized_instruction(raw_best_prompt: str, fallback_instruction: str) -> str:
    text = (raw_best_prompt or "").strip()
    if not text:
        return fallback_instruction.strip()

    marker_positions = []
    for marker in ["\n[Question]", "\n  [Question]", "\nQuestion:"]:
        pos = text.find(marker)
        if pos != -1:
            marker_positions.append(pos)
    if marker_positions:
        text = text[: min(marker_positions)].strip()

    answer_format_pos = text.lower().find("restituisci solo json valido nel formato")
    if answer_format_pos != -1:
        text = text[:answer_format_pos].strip()

    return text or fallback_instruction.strip()


def compose_pipeline_prompt(optimized_instruction: str) -> str:
    return (
        f"{optimized_instruction}\n\n"
        f"{PIPELINE_OUTPUT_INSTRUCTION}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Candidate answer:\n"
        "{candidate_answer}\n\n"
        "Ground truth answers:\n"
        "{ground_truth_answers}\n"
    )


def write_setup_config(path: Path, args: argparse.Namespace) -> None:
    content = f"""assistant_llm:
  prompt_opt: promptopt-model
dir_info:
  base_dir: logs
  log_dir_name: glue_logs
experiment_name: {args.experiment_name}
mode: offline
description: prompt optimization for MAIA LLM-as-a-Judge Italian
"""
    path.write_text(content, encoding="utf-8")


def ensure_promptwizard_log_dirs(experiment_name: str) -> None:
    """Create PromptWizard log folders that ParamLogger expects to exist."""
    base = Path("logs") / experiment_name
    (base / "io_logs").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "glue_logs").mkdir(parents=True, exist_ok=True)


def ensure_promptwizard_import(promptwizard_root: str) -> None:
    root = Path(promptwizard_root)
    if not root.exists():
        raise FileNotFoundError(f"PromptWizard path not found: {promptwizard_root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def patch_promptwizard_temperature_handling() -> None:
    """Monkey patch PromptWizard API caller to tolerate models that reject temperature=0.0."""
    import promptwizard.glue.common.llm.llm_mgr as llm_mgr

    def _create_with_temperature_fallback(client, request_kwargs):
        from openai import BadRequestError

        try:
            return client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "temperature" not in message:
                raise

            no_temp = dict(request_kwargs)
            no_temp.pop("temperature", None)
            try:
                return client.chat.completions.create(**no_temp)
            except BadRequestError:
                default_temp = dict(request_kwargs)
                default_temp["temperature"] = 1
                return client.chat.completions.create(**default_temp)

    def patched_call_api(messages):
        from openai import OpenAI
        from openai import AzureOpenAI
        from azure.identity import AzureCliCredential, get_bearer_token_provider

        if os.environ["USE_OPENAI_API_KEY"] == "True":
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = _create_with_temperature_fallback(
                client,
                {
                    "model": os.environ["OPENAI_MODEL_NAME"],
                    "messages": messages,
                    "temperature": 0.0,
                },
            )
        else:
            token_provider = get_bearer_token_provider(
                AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client = AzureOpenAI(
                api_version=os.environ["OPENAI_API_VERSION"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_ad_token_provider=token_provider,
            )
            response = _create_with_temperature_fallback(
                client,
                {
                    "model": os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    "messages": messages,
                    "temperature": 0.0,
                },
            )

        return response.choices[0].message.content

    llm_mgr.call_api = patched_call_api


def patch_promptwizard_refinement_format_handling() -> None:
    """Monkey patch PromptWizard refinement to avoid hard crash on malformed tag output."""
    from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
    from promptwizard.glue.promptopt.techniques.critique_n_refine import core_logic

    def _sanitize_refined_text(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^```[a-zA-Z]*\\n", "", cleaned)
        cleaned = re.sub(r"\\n```$", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip("'")
        return cleaned

    def patched_critique_and_refine(self, prompt, critique_example_set, further_enhance=False):
        example_string = self.data_processor.collate_to_str(
            critique_example_set, self.prompt_pool.quest_reason_ans
        )

        meta_critique_prompt = (
            self.prompt_pool.meta_positive_critique_template
            if further_enhance
            else self.prompt_pool.meta_critique_template
        )
        meta_critique_prompt = meta_critique_prompt.format(
            instruction=prompt, examples=example_string
        )

        critique_text = self.chat_completion(meta_critique_prompt, self.prompt_pool.expert_profile)
        critique_refine_prompt = self.prompt_pool.critique_refine_template.format(
            instruction=prompt,
            examples=example_string,
            critique=critique_text,
            steps_per_sample=1,
        )

        last_output = ""
        for attempt in range(3):
            refine_prompt = critique_refine_prompt
            if attempt > 0:
                refine_prompt += (
                    "\nIMPORTANT: Output exactly one improved prompt wrapped as "
                    "<START> ... </END>. Do not output any extra text."
                )

            raw_refined = self.chat_completion(refine_prompt, self.prompt_pool.expert_profile)
            last_output = raw_refined or ""

            tagged = re.findall(
                DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, raw_refined or ""
            )
            if tagged:
                return tagged[0].strip()

        fallback = _sanitize_refined_text(last_output)
        if fallback:
            self.logger.warning(
                "PromptWizard returned refinement without expected tags; using sanitized fallback text."
            )
            return fallback

        raise ValueError("Prompt refinement failed after retries: empty/malformed output.")

    core_logic.CritiqueNRefine.critique_and_refine = patched_critique_and_refine


def main() -> None:
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    ensure_promptwizard_import(args.promptwizard_root)
    patch_promptwizard_temperature_handling()
    patch_promptwizard_refinement_format_handling()

    from promptwizard.glue.promptopt.instantiate import GluePromptOpt
    from .processor import MaiaJudgeProcessing

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing. Set env var or pass --openai-api-key.")

    os.environ["USE_OPENAI_API_KEY"] = "True"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL_NAME"] = args.openai_model
    os.environ.setdefault("MODEL_TYPE", "AzureOpenAI")

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    config_dir = work_dir / "configs"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    examples = build_judge_examples(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        split=args.split,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    if len(examples) < 4:
        raise RuntimeError("Not enough examples generated to run train/test optimization.")

    train_rows, test_rows = split_train_test(examples, train_ratio=args.train_ratio, seed=args.seed)
    if not train_rows or not test_rows:
        raise RuntimeError("Train/test split produced an empty partition.")

    processor = MaiaJudgeProcessing()
    train_jsonl = data_dir / "train.jsonl"
    test_jsonl = data_dir / "test.jsonl"
    processor.dataset_to_jsonl(str(train_jsonl), dataset=train_rows)
    processor.dataset_to_jsonl(str(test_jsonl), dataset=test_rows)

    seen_set_size = min(args.seen_set_size, len(train_rows))
    few_shot_count = min(args.few_shot_count, seen_set_size)
    if args.scenario == 1:
        few_shot_count = 0

    promptopt_config_path = config_dir / "promptopt_config.yaml"
    setup_config_path = config_dir / "setup_config.yaml"
    write_promptopt_config(promptopt_config_path, args, seen_set_size=seen_set_size, few_shot_count=few_shot_count)
    write_setup_config(setup_config_path, args)

    if args.verbose:
        print(f"Scenario: {args.scenario}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Generated examples: {len(examples)}")
        print(f"Train examples: {len(train_rows)}")
        print(f"Test examples: {len(test_rows)}")
        print(f"Seen set size: {seen_set_size}")
        print(f"Few-shot count: {few_shot_count}")
        print(f"Work dir: {work_dir}")

    best_prompt = ""
    expert_profile = ""
    test_accuracy = None
    synthetic_train_jsonl = work_dir / "train_synthetic.jsonl"

    if args.scenario == 3:
        ensure_promptwizard_log_dirs(args.experiment_name)
        gp = GluePromptOpt(
            str(promptopt_config_path),
            str(setup_config_path),
            dataset_jsonl=str(train_jsonl),
            data_processor=processor,
        )
        best_prompt, expert_profile = gp.get_best_prompt(
            use_examples=True,
            run_without_train_examples=False,
            generate_synthetic_examples=False,
        )
        test_accuracy = gp.evaluate(str(test_jsonl))

    elif args.scenario == 2:
        # Phase 1: no train data, generate synthetic examples.
        gp_synth = GluePromptOpt(
            str(promptopt_config_path),
            str(setup_config_path),
            dataset_jsonl=None,
            data_processor=None,
        )
        with contextlib.chdir(work_dir):
            ensure_promptwizard_log_dirs(args.experiment_name)
            gp_synth.get_best_prompt(
                use_examples=False,
                run_without_train_examples=False,
                generate_synthetic_examples=True,
            )

        if not synthetic_train_jsonl.exists():
            raise RuntimeError(
                f"Synthetic dataset was not generated at expected path: {synthetic_train_jsonl}"
            )

        # Phase 2: optimize using synthetic train set and include in-context examples.
        ensure_promptwizard_log_dirs(args.experiment_name)
        gp = GluePromptOpt(
            str(promptopt_config_path),
            str(setup_config_path),
            dataset_jsonl=str(synthetic_train_jsonl),
            data_processor=processor,
        )
        best_prompt, expert_profile = gp.get_best_prompt(
            use_examples=True,
            run_without_train_examples=False,
            generate_synthetic_examples=False,
        )
        test_accuracy = gp.evaluate(str(test_jsonl))

    else:
        # Scenario 1: no train data and no in-context examples.
        ensure_promptwizard_log_dirs(args.experiment_name)
        gp = GluePromptOpt(
            str(promptopt_config_path),
            str(setup_config_path),
            dataset_jsonl=None,
            data_processor=None,
        )
        best_prompt, expert_profile = gp.get_best_prompt(
            use_examples=False,
            run_without_train_examples=True,
            generate_synthetic_examples=False,
        )

    raw_best_prompt = (best_prompt or "").strip()
    optimized_instruction = extract_optimized_instruction(
        raw_best_prompt,
        fallback_instruction=args.base_instruction,
    )
    pipeline_ready_prompt = compose_pipeline_prompt(optimized_instruction)

    (work_dir / "best_prompt_raw.txt").write_text(raw_best_prompt, encoding="utf-8")
    (work_dir / "best_prompt.txt").write_text(pipeline_ready_prompt, encoding="utf-8")
    (work_dir / "expert_profile.txt").write_text(str(expert_profile or ""), encoding="utf-8")

    summary = {
        "scenario": args.scenario,
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "split": args.split,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "source_example_count": len(examples),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "seen_set_size": seen_set_size,
        "few_shot_count": few_shot_count,
        "openai_model": args.openai_model,
        "synthetic_train_jsonl": str(synthetic_train_jsonl) if synthetic_train_jsonl.exists() else None,
        "test_accuracy": test_accuracy,
        "promptopt_config_path": str(promptopt_config_path),
        "setup_config_path": str(setup_config_path),
        "train_jsonl": str(train_jsonl),
        "test_jsonl": str(test_jsonl),
        "best_prompt_path": str(work_dir / "best_prompt.txt"),
        "best_prompt_raw_path": str(work_dir / "best_prompt_raw.txt"),
        "expert_profile_path": str(work_dir / "expert_profile.txt"),
    }
    save_jsonl(work_dir / "run_summary.jsonl", [summary])
    (work_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if test_accuracy is None:
        print("Optimization completed. Test accuracy not available for this scenario.")
    else:
        print(f"Optimization completed. Test accuracy: {test_accuracy:.4f}")
    print(f"Best prompt saved to: {work_dir / 'best_prompt.txt'}")
    print(f"Summary saved to: {work_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
