import re
import json
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.llm import HFLLMConfig, load_hf_chat_llm
from agent.llm import VLLMConfig, load_vllm_chat_llm

def parse_args():
    parser = ArgumentParser(description="Rephrase questions with mask replacement.")
    parser.add_argument('--test', action='store_true', help='Use test set instead of val set')
    parser.add_argument('--quantization', type=str, default='none', choices=['none', '4bit', '8bit'],
                        help='Quantization mode: none (full precision), 4bit, or 8bit')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='HF model name or local path (defaults to <repo>/Qwen2.5-7B-Instruct)',
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='HF tokenizer name or local path (defaults to --model)',
    )
    parser.add_argument('--revision', type=str, default=None, help='Model/tokenizer revision')
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Allow custom model code from the Hugging Face Hub',
    )
    parser.add_argument(
        '--device_map',
        type=str,
        default=None,
        help='Device map for transformers (e.g. cuda:0, cpu, auto)',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        help='Model dtype: auto, fp16, bf16, fp32',
    )
    parser.add_argument(
        '--llm_backend',
        type=str,
        default='hf',
        choices=['hf', 'vllm'],
        help='LLM backend: hf (transformers) or vllm',
    )
    parser.add_argument(
        '--vllm_tensor_parallel_size',
        type=int,
        default=1,
        help='vLLM tensor parallel size (only for --llm_backend vllm)',
    )
    parser.add_argument(
        '--vllm_gpu_memory_utilization',
        type=float,
        default=0.9,
        help='vLLM GPU memory utilization (only for --llm_backend vllm)',
    )
    parser.add_argument(
        '--vllm_max_model_len',
        type=int,
        default=None,
        help='vLLM max model length (only for --llm_backend vllm)',
    )
    return parser.parse_args()

def load_model(args):
    default_model_path = os.path.join(REPO_ROOT, "Qwen2.5-7B-Instruct")
    model_name_or_path = args.model or default_model_path
    backend = getattr(args, "llm_backend", "hf")
    print(f"Loading model: {model_name_or_path} (backend={backend})")

    if backend == "vllm":
        if (args.quantization or "none").strip().lower() != "none":
            raise ValueError("vLLM backend currently requires --quantization none.")
        return load_vllm_chat_llm(
            VLLMConfig(
                model_name_or_path=model_name_or_path,
                tokenizer_name_or_path=args.tokenizer,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
            )
        )

    return load_hf_chat_llm(
        HFLLMConfig(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=args.tokenizer,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
            dtype=args.dtype,
            quantization=args.quantization,
        )
    )

# This will be initialized in main
llm = None

TYPE_PROMPT = open(os.path.join(REPO_ROOT, 'agent', 'prompt', 'rephrase.txt'), 'r').read()

OBJECT_TYPES = ('pallet', 'transporter', 'shelf', 'buffer')
OBJECT_TYPE_RE = re.compile(r"(pallet|transporter|shelf|buffer)", re.IGNORECASE)

def _extract_object_type(text: str):
    match = OBJECT_TYPE_RE.search(text)
    if not match:
        return None
    return match.group(1).lower()

def _mark_nth_mask(question: str, mask_index: int, marker: str = "<target_mask>") -> str:
    matches = list(re.finditer(r"<mask>", question))
    if mask_index < 0 or mask_index >= len(matches):
        raise ValueError(
            f"mask_index {mask_index} out of range; found {len(matches)} '<mask>' tokens"
        )
    start, end = matches[mask_index].span()
    return question[:start] + marker + question[end:]

def predict_object_type(question_with_target_mask: str) -> str:
    question_with_target_mask = question_with_target_mask.replace('<image>\n', '')
    if "<target_mask>" not in question_with_target_mask:
        raise ValueError("predict_object_type expects '<target_mask>' in the input question")

    prompt_text = TYPE_PROMPT.replace("<input>", question_with_target_mask)
    strict_prompt_text = (
        "Return exactly one word from: pallet, transporter, shelf, buffer.\n"
        f"Sentence: {question_with_target_mask}\n"
        "Output:"
    )

    last_response = None
    for user_content in (prompt_text, strict_prompt_text):
        messages = [
            {"role": "system", "content": "You are a strict classifier. Reply with a single word."},
            {"role": "user", "content": user_content},
        ]
        last_response = llm.generate(
            messages,
            max_new_tokens=5,
            do_sample=False,
        ).strip()

        object_type = _extract_object_type(last_response)
        if object_type in OBJECT_TYPES:
            return object_type

    raise ValueError(
        "Could not extract a valid object type from model output. "
        f"Expected one of {OBJECT_TYPES}, got: {last_response!r}"
    )

def replace_masks_with_objects(original_question: str) -> str:
    original_question = original_question.replace('<image>\n', '')

    object_keywords = ['shelf', 'transporter', 'pallet', 'buffer']

    # Tokenize while keeping <mask> as a single token, words, and punctuation
    token_pattern = re.compile(r'<mask>|[a-zA-Z]+|[^\s\w]')
    tokens = token_pattern.findall(original_question)

    object_counters = {obj: 0 for obj in object_keywords}
    modified_tokens = []
    last_object = None
    last_object_updated = False
    mask_index = -1

    for i, token in enumerate(tokens):
        if token == '<mask>':
            mask_index += 1
            
            if not last_object_updated:
                marked_question = _mark_nth_mask(original_question, mask_index)
                last_object = predict_object_type(marked_question)
                last_object_updated = True

            replacement = f"<{last_object}_{object_counters[last_object]}>"
            object_counters[last_object] += 1
            modified_tokens.append(replacement)
        else:
            if i > 0 and tokens[i - 1] == '<mask>':
                last_object_updated = False
            if token.lower() == 'shelves':
                token_lower = 'shelf'
            else:
                token_lower = token.lower().rstrip('s')
            if token_lower in object_keywords:
                last_object = token_lower
                last_object_updated = True
            modified_tokens.append(token)

    # Rebuild the sentence with appropriate spacing
    modified_question = ''
    prev_token = ''
    for token in modified_tokens:
        if prev_token:
            if (re.match(r'[a-zA-Z0-9>]', prev_token) and re.match(r'[a-zA-Z0-9<]', token)):
                modified_question += ' '
            elif re.match(r'[>)]', prev_token) and re.match(r'[A-Za-z<]', token):
                modified_question += ' '
            elif re.match(r'[A-Za-z0-9]', prev_token) and re.match(r'[<]', token):
                modified_question += ' '
        modified_question += token
        prev_token = token

    modified_question = ' '.join(modified_tokens)

    return modified_question


if __name__ == "__main__":
    args = parse_args()
    
    # Load model with quantization option
    llm = load_model(args)

    if args.test:
        input_path = 'data/test/test.json'
        output_path = 'data/test/rephrased_test.json'
    else:
        input_path = 'data/val/val.json'
        output_path = 'data/val/rephrased_val.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    
    for item in tqdm(data, desc="Processing JSON data"):
        question_id = item.get('id')
        rephrased_conversations = []
        for conversation in item.get('conversations', []):
            if conversation.get('from') == 'human':
                original_question = conversation.get('value')
                if original_question:
                    rephrased_question = replace_masks_with_objects(original_question)
                    rephrased_conversations.append({
                        "from": "human",
                        "value": rephrased_question
                    })
                else:
                    rephrased_conversations.append(conversation)
            else:
                rephrased_conversations.append(conversation)
        item['rephrase_conversations'] = rephrased_conversations
        processed_data.append(item)

    
    with open(output_path, 'w') as out_file:
        json.dump(processed_data, out_file, indent=4)
