"""Audit pinned Qwen tokenization and build assistant-only loss labels."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform

MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
SYSTEM_PROMPT = 'Ты помощник по вопросам дома и квартиры. Отвечай на русском языке, понятно и по существу.'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def encode_training(row, tokenizer, max_length=1024):
    messages = row['messages']
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError(f"Unexpected roles in {row['id']}")
    if messages[0]['content'] != SYSTEM_PROMPT:
        raise ValueError(f"Unexpected system prompt in {row['id']}")
    for message in messages:
        if any(marker in message['content'] for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
            raise ValueError(f"Chat control token in content: {row['id']}")
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError(f"Prompt is not a token prefix: {row['id']}")
    if len(full) > max_length:
        raise ValueError(f"Sequence exceeds {max_length}: {row['id']} ({len(full)})")
    target = full[len(prompt):]
    if not target or tokenizer.eos_token_id not in target:
        raise ValueError(f"Missing answer/EOS: {row['id']}")
    # Include answer and its end-of-turn token, mask system and user completely.
    labels = [-100] * len(prompt) + target
    return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory must be new')
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=args.cache_dir)
    measurements = []
    for row in read_jsonl(args.data_dir / 'train.jsonl'):
        encoded = encode_training(row, tokenizer, args.max_length)
        target_length = sum(label != -100 for label in encoded['labels'])
        measurements.append({'id': row['id'], 'split': 'train', 'prompt_tokens': len(encoded['input_ids'])-target_length, 'target_tokens': target_length, 'total_tokens': len(encoded['input_ids'])})
    for split in ('validation', 'test'):
        for row in read_jsonl(args.data_dir / f'{split}_questions.jsonl'):
            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': row['question']}], tokenize=True, add_generation_prompt=True)
            if len(prompt) + 512 > args.max_length:
                raise ValueError(f"Evaluation prompt lacks room for 512 new tokens: {row['id']}")
            measurements.append({'id': row['id'], 'split': split, 'prompt_tokens': len(prompt), 'target_tokens': 0, 'total_tokens': len(prompt)})
    summary = {'model_id': MODEL_ID, 'revision': REVISION, 'python': platform.python_version(), 'transformers': transformers.__version__, 'max_length': args.max_length, 'truncation': False, 'label_rule': 'Only assistant answer plus end-of-turn suffix; prompt labels -100.', 'counts': dict(Counter(r['split'] for r in measurements)), 'splits': {}, 'sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data_dir.glob('*.jsonl'))}}
    for split in ('train', 'validation', 'test'):
        lengths = sorted(r['total_tokens'] for r in measurements if r['split'] == split)
        summary['splits'][split] = {'min_tokens': min(lengths), 'max_tokens': max(lengths), 'mean_tokens': sum(lengths)/len(lengths)}
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'token_lengths.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (args.output_dir / 'chat_template.jinja').write_text(tokenizer.chat_template)
    sample = read_jsonl(args.data_dir / 'train.jsonl')[0]
    encoded = encode_training(sample, tokenizer, args.max_length)
    (args.output_dir / 'mask_example.json').write_text(json.dumps({'id':sample['id'], 'masked_prompt':tokenizer.decode([t for t,l in zip(encoded['input_ids'],encoded['labels']) if l == -100]), 'supervised_text':tokenizer.decode([l for l in encoded['labels'] if l != -100])}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
