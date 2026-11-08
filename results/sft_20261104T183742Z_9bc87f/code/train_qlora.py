"""Train and reload a QLoRA adapter on a single Colab T4.

Run from the repository root. Each run requires a new output directory.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from prepare_sft import MODEL_ID, REVISION, encode_training, read_jsonl


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['sanity', 'train'], default='sanity')
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error('Epochs and learning rate must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status':'running', 'mode':args.mode, 'started_at':datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / 'run.json', status)
    try:
        import torch
        import transformers
        import peft
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required. Select T4 in Colab.')
        manifest = json.loads((args.data_dir / 'manifest.json').read_text())
        for filename, expected in manifest['sha256'].items():
            actual = hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Data changed since manifest: {filename}')
        rows = read_jsonl(args.data_dir / 'train.jsonl')
        if args.mode == 'sanity':
            # One accepted pilot example from each topic.
            rows = [rows[i] for i in (0,3,6,9,12,15,18,21)]
        config = {'model_id':MODEL_ID, 'revision':REVISION, 'seed':42, 'mode':args.mode, 'examples':len(rows), 'review_status_counts':dict(Counter(r['review_status'] for r in rows)), 'max_length':1024, 'learning_rate':args.learning_rate, 'epochs':args.epochs, 'max_steps':2 if args.mode == 'sanity' else -1, 'batch_size':1, 'gradient_accumulation_steps':8, 'lora_r':16, 'lora_alpha':32, 'lora_dropout':0.05, 'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], 'quantization':'NF4, double quantization, FP16 compute', 'loss':'assistant answer plus end-of-turn suffix', 'data_manifest':manifest, 'train_ids':[r['id'] for r in rows]}
        write_json(args.output_dir / 'config.json', config)
        (args.output_dir / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        (args.output_dir / 'nvidia-smi.txt').write_text(gpu.stdout + gpu.stderr)
        write_json(args.output_dir / 'environment.json', {'python':platform.python_version(), 'torch':torch.__version__, 'cuda':torch.version.cuda, 'transformers':transformers.__version__, 'peft':peft.__version__, 'gpu':torch.cuda.get_device_name(0), 'gpu_bytes':torch.cuda.get_device_properties(0).total_memory})
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        tokenizer.padding_side = 'right'
        encoded = [encode_training(row, tokenizer, 1024) for row in rows]
        dataset = Dataset.from_list(encoded)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')
        probe = collator([dict(encoded[0]),dict(encoded[-1])])
        assert torch.all(probe['labels'][probe['attention_mask']==0] == -100)
        for i, item in enumerate((encoded[0], encoded[-1])):
            assert probe['labels'][i,:len(item['labels'])].tolist() == item['labels']
        write_json(args.output_dir / 'mask_check.json', {'passed':True, 'batch_shape':list(probe['input_ids'].shape), 'target_token_counts':[sum(t != -100 for t in item['labels']) for item in encoded], 'truncation':False})
        del probe
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, quantization_config=quantization, torch_dtype=torch.float16, device_map={'':0}, attn_implementation='sdpa')
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False})
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=config['target_modules'], bias='none', task_type='CAUSAL_LM'))
        model.peft_config['default'].revision = REVISION
        trainable, total = model.get_nb_trainable_parameters()
        write_json(args.output_dir / 'parameters.json', {'trainable':trainable, 'total':total, 'fraction':trainable/total})
        model.print_trainable_parameters()

        class FiniteLog(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ('loss','grad_norm'):
                    if key in (logs or {}) and not math.isfinite(float(logs[key])):
                        raise RuntimeError(f'Non-finite {key} at step {state.global_step}')
                with (Path(args.output_dir).parent / 'training-log.jsonl').open('a') as handle:
                    handle.write(json.dumps({'step':state.global_step, **(logs or {})})+'\n')

        training_args = TrainingArguments(output_dir=str(args.output_dir/'checkpoints'), num_train_epochs=args.epochs, max_steps=config['max_steps'], per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=args.learning_rate, lr_scheduler_type='linear', warmup_ratio=0.1, weight_decay=0.0, max_grad_norm=1.0, optim='adamw_torch', fp16=True, bf16=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False}, logging_steps=1, logging_nan_inf_filter=False, save_strategy='epoch', save_total_limit=2, eval_strategy='no', report_to='none', seed=42, data_seed=42, dataloader_num_workers=0, remove_unused_columns=False)
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator, processing_class=tokenizer, callbacks=[FiniteLog()])
        torch.cuda.reset_peak_memory_stats()
        training_started = time.monotonic()
        result = trainer.train()
        if not math.isfinite(result.training_loss) or result.global_step < 1:
            raise RuntimeError('Training did not produce finite loss and optimizer steps')
        write_json(args.output_dir/'training_metrics.json', {**result.metrics, 'elapsed_seconds':time.monotonic()-training_started, 'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        adapter_dir = args.output_dir/'adapter'
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        trainer.state.save_to_json(str(args.output_dir/'trainer_state.json'))
        # Release training allocations before reloading the saved artifact.
        del trainer, model, dataset, encoded, collator
        gc.collect()
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, quantization_config=quantization, torch_dtype=torch.float16, device_map={'':0}, attn_implementation='sdpa')
        reloaded = PeftModel.from_pretrained(base, adapter_dir)
        reloaded.eval()
        reloaded.config.use_cache = True
        input_ids = tokenizer.apply_chat_template(rows[0]['messages'][:2], add_generation_prompt=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = reloaded.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=512, do_sample=False, num_beams=1, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        suffix = generated[0,input_ids.shape[1]:].tolist()
        answer = tokenizer.decode(suffix, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError('Reloaded adapter produced an empty response')
        write_json(args.output_dir/'reload_probe.json', {'id':rows[0]['id'], 'question':rows[0]['messages'][1]['content'], 'answer':answer, 'new_tokens':len(suffix), 'ended_with_eos':bool(suffix and suffix[-1] == tokenizer.eos_token_id), 'purpose':'Artifact reload check on a training question; excluded from quality evaluation.'})
        write_json(args.output_dir/'adapter_manifest.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(adapter_dir.iterdir()) if p.is_file()})
        status.update(status='completed', optimizer_steps=result.global_step, reload_verified=True)
    except Exception:
        status.update(status='failed')
        (args.output_dir/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir/'run.json', status)


if __name__ == '__main__':
    main()
