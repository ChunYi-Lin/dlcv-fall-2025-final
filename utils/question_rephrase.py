import re
import json
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def parse_args():
    parser = ArgumentParser(description="Rephrase questions with mask replacement.")
    parser.add_argument('--test', action='store_true', help='Use test set instead of val set')
    parser.add_argument('--quantization', type=str, default='none', choices=['none', '4bit', '8bit'],
                        help='Quantization mode: none (full precision), 4bit, or 8bit')
    return parser.parse_args()

def load_model(quantization='none'):
    MODEL_PATH = "./Qwen2.5-7B-Instruct" 

    print(f"Loading from local path: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if quantization == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="cuda:0"
        )
    elif quantization == '8bit':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="cuda:0"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="cuda:0"
        )
    
    print(f"Model loaded with quantization: {quantization}")
    return model, tokenizer

# These will be initialized in main
model = None
tokenizer = None

prompt = open('agent/prompt/rephrase.txt', 'r').read()

def verify(original_question, rephrased_question: str) -> bool:
    original_count = original_question.count('<') + original_question.count('>')
    rephrased_count = rephrased_question.count('<') + rephrased_question.count('>')
    return original_count == rephrased_count

def rephrase_question(question: str) -> str:
    question = question.replace('<image>\n', '')
    input_text = prompt.replace('<input>', question)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

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

    for i, token in enumerate(tokens):
        if token == '<mask>':
            
            if not last_object_updated:
                rephrase = rephrase_question(original_question)
                print(f"Original question: {original_question}")
                print(f"Rephrased question: {rephrase}")
                if not verify(original_question, rephrase):
                    import pdb; pdb.set_trace()
                return rephrase   

            replacement = f"<{last_object}_{object_counters[last_object]}>"
            object_counters[last_object] += 1
            modified_tokens.append(replacement)
        else:
            if tokens[i - 1] == '<mask>':
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
    model, tokenizer = load_model(args.quantization)

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
