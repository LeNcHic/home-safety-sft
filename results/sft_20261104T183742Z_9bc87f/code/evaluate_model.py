"""Generate held-out answers with the frozen baseline settings on a Colab T4.

Run base and adapter in separate processes and new output directories. Partial
answers and failure details are retained if generation fails.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from prepare_sft import MODEL_ID, REVISION, SYSTEM_PROMPT, read_jsonl


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/v1'))
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    questions_path = args.data_dir / f'{args.split}_questions.jsonl'
    for filename, expected in manifest['sha256'].items():
        if digest(args.data_dir / filename) != expected:
            parser.error(f'Data changed since manifest: {filename}')
    questions = read_jsonl(questions_path)
    if not questions or len({row['id'] for row in questions}) != len(questions):
        parser.error('Questions must be nonempty with unique IDs')
    adapter_hashes = None
    if args.adapter:
        adapter_config = json.loads((args.adapter / 'adapter_config.json').read_text())
        if (adapter_config['base_model_name_or_path'] != MODEL_ID
                or adapter_config['revision'] != REVISION):
            parser.error('Adapter base model or revision differs from frozen baseline')
        adapter_hashes = {name: digest(args.adapter / name) for name in
                          ('adapter_config.json', 'adapter_model.safetensors')}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(questions_path, args.output_dir / 'questions.jsonl')
    shutil.copy2(__file__, args.output_dir / 'evaluate_model.py')
    config = {
        'model_id': MODEL_ID, 'model_revision': REVISION, 'seed': 42,
        'system_prompt': SYSTEM_PROMPT, 'split': args.split,
        'variant': 'sft' if args.adapter else 'base',
        'adapter_sha256': adapter_hashes, 'questions_sha256': digest(questions_path),
        'quantization': {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                         'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'float16'},
        'dtype': 'float16', 'attention_implementation': 'sdpa',
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 512,
                       'repetition_penalty': 1.0},
    }
    write_json(args.output_dir / 'config.json', config)
    status = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'expected_answers': len(questions), 'completed_answers': 0}
    write_json(args.output_dir / 'run.json', status)
    started = time.monotonic()
    try:
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, set_seed
        from peft import PeftModel
        if not torch.cuda.is_available():
            raise RuntimeError('Select a T4 GPU in Colab')
        set_seed(config['seed'])
        write_json(args.output_dir / 'environment.json', {
            'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'transformers': transformers.__version__,
            'peft': peft.__version__, 'gpu': torch.cuda.get_device_name(0),
        })
        (args.output_dir / 'pip-freeze.txt').write_text(
            subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        (args.output_dir / 'nvidia-smi.txt').write_text(
            subprocess.check_output(['nvidia-smi'], text=True))
        torch.cuda.reset_peak_memory_stats()
        load_started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
        quantization_options = {**config['quantization'], 'bnb_4bit_compute_dtype': torch.float16}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, quantization_config=BitsAndBytesConfig(**quantization_options),
            torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa',
            use_safetensors=True, trust_remote_code=False,
        )
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        model.config.use_cache = True
        status['model_load_seconds'] = time.monotonic() - load_started
        generation = GenerationConfig(
            **config['generation'], eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        )
        write_json(args.output_dir / 'generation_config.json', generation.to_dict())
        model.config.to_json_file(args.output_dir / 'model_config.json')
        (args.output_dir / 'chat_template.jinja').write_text(tokenizer.get_chat_template())
        eos_ids = generation.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        with (args.output_dir / 'answers.jsonl').open('x', encoding='utf-8') as answers_file:
            for row in questions:
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                            {'role': 'user', 'content': row['question']}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda:0')
                input_tokens = inputs['input_ids'].shape[-1]
                torch.cuda.synchronize()
                question_started = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, generation_config=generation)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - question_started
                answer_ids = output[0, input_tokens:].tolist()
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
                ended = bool(answer_ids) and answer_ids[-1] in eos_ids
                record = {key: row[key] for key in ('id', 'family_id', 'split', 'topic', 'risk', 'question')}
                record.update(model_id=MODEL_ID, model_revision=REVISION, variant=config['variant'],
                              messages=messages, rendered_prompt=rendered, answer=answer,
                              generated_token_ids=answer_ids, input_tokens=input_tokens,
                              generated_tokens=len(answer_ids), generation_seconds=elapsed,
                              ended_with_eos=ended, hit_token_limit=len(answer_ids) >= 512 and not ended)
                answers_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                answers_file.flush()
                status['completed_answers'] += 1
                write_json(args.output_dir / 'run.json', status)
                print(f"{config['variant']} {status['completed_answers']}/{len(questions)} {row['id']} tokens={len(answer_ids)} eos={ended}", flush=True)
        status.update(status='completed', answers_sha256=digest(args.output_dir / 'answers.jsonl'),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException:
        status['status'] = 'failed'
        (args.output_dir / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        status.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir / 'run.json', status)


if __name__ == '__main__':
    main()
