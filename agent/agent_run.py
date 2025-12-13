import re
import json
import os
import ast
from tools import tools_api
from argparse import ArgumentParser
from tqdm import tqdm
from mask import parse_masks_from_conversation
import torch

try:
    from llm import HFLLMConfig, load_hf_chat_llm
except ImportError:
    from agent.llm import HFLLMConfig, load_hf_chat_llm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

def convs_output_path_for(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}_convs{ext or '.json'}"

def parse_args():
    parser = ArgumentParser(description="Agent for answering questions with inside model.")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to run on')
    parser.add_argument('--json_path', type=str, default=None,
                        help='Path to input JSON (defaults to data/<split>/rephrased_<split>.json)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Image directory (defaults to data/<split>/images)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the results (defaults to output/<split>.json)')
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
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Max new tokens per turn')
    parser.add_argument('--do_sample', action='store_true', help='Enable sampling for generation')
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (requires --do_sample)',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=None,
        help='Nucleus sampling p (requires --do_sample)',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Top-k sampling (requires --do_sample)',
    )
    return parser.parse_args()

class Agent:
    def __init__(
        self,
        llm,
        tools_api,
        input,
        image_dir: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        self.llm = llm
        self.tools_api = tools_api
        self.messages = []
        self.conversation = []
        self.prompt_preamble = open(os.path.join(THIS_DIR, 'prompt', 'agent_example.txt'), 'r').read()
        self.answer_preamble = open(os.path.join(THIS_DIR, 'prompt', 'answer.txt'), 'r').read()
        self.input = input
        self.image_dir = image_dir
        self.masks = None
        self.question = None
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate_response(self, new_user_content):
        # Append new user message to history
        self.messages.append({"role": "user", "content": new_user_content})

        response_text = self.llm.generate(
            self.messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        # Append assistant response to history
        self.messages.append({"role": "assistant", "content": response_text})
        
        return response_text

    def set_masks(self):
        conversation = self.input['rephrase_conversations'][0]['value']
        rle_data = self.input['rle']
        self.masks = parse_masks_from_conversation(conversation, rle_data)
        self.tools_api.update_masks(self.masks)
        if not self.masks:
            raise ValueError("No valid masks found in the conversation.")
    
    def format_answer(self):
        answer = self.generate_response(self.answer_preamble)
        answer = answer.strip()

        if answer in self.masks:
            return self.masks[answer].region_id
        else:
            return answer

    def set_question(self):
        self.messages = [] # Reset history
        self.question = self.input['rephrase_conversations'][0]['value']
        full_prompt = self.prompt_preamble.replace("<question>", self.question)
        self.messages.append({"role": "system", "content": "You are a helpful agent."})
        self.tools_api.update_image(os.path.join(self.image_dir, self.input['image']))
        
        # Call loop passing the initial prompt as the first "trigger"
        return self._conversation_loop(initial_prompt=full_prompt)

    def _conversation_loop(self, budget=10, initial_prompt=None):
        usage = 0
        execute_flag = False
        
        # Handle the very first message
        current_input = initial_prompt
        
        while usage < budget:
            usage += 1
            
            # Generate
            assistant_text = self.generate_response(current_input)
            
            # Check for <execute>
            execute_match = re.search(r"<execute>(.*?)</execute>", assistant_text, re.DOTALL)
            if execute_match:
                execute_flag = True
                command = execute_match.group(1).strip()
                try:
                    result = self._execute_function(command)
                    # The result becomes the input for the next turn
                    current_input = f"{result}"
                except Exception as e:
                    current_input = f"Error: {str(e)}"
                continue

            # Check for <answer>
            answer_match = re.search(r"<answer>(.*?)</answer>", assistant_text, re.DOTALL)
            if answer_match: # Removed 'and execute_flag' strict check if you want it to be more robust
                return answer_match.group(1).strip()

            # If no tags, maybe provide a hint or break
            print("No valid action found.")
            break

    def _execute_function(self, command):
        """Parses and executes a function call from Gemini."""
        command = command.strip().replace('\\n', '').replace('\n', '').replace('\r', '').replace('<', '').replace('>', '')
        match = re.match(r"(\w+)\s*\((.*)\)", command)
        if not match:
            raise ValueError(f"Invalid function call: {command}")

        func_name, args_str = match.groups()

        func_map = {
            "dist": self.tools_api.dist,
            "closest": self.tools_api.closest,
            "is_left": self.tools_api.is_left,
            "is_right": self.tools_api.is_right,
            "inside": self.tools_api.inside,
            "most_right": self.tools_api.most_right,
            "most_left": self.tools_api.most_left,
            "middle": self.tools_api.middle,
            "is_empty": self.tools_api.is_empty
        }

        if func_name not in func_map:
            raise ValueError(f"Unknown function: {func_name}")

        parsed_args = self._parse_arguments(args_str)
        result = func_map[func_name](*parsed_args)
        return result

    def _parse_arguments(self, args_str):
        """
        Parses and resolves function arguments from a string.
        Supports nested lists and variable references like <buffer_1>.
        """

        def resolve(node):
            if isinstance(node, ast.Name):
                key = node.id
                if key.startswith('<') and key.endswith('>'):
                    key = key[1:-1]
                if key not in self.masks:
                    raise ValueError(f"Mask '{key}' not found in the provided masks.")
                return self.masks[key]
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.List):
                return [resolve(elt) for elt in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(resolve(elt) for elt in node.elts)
            elif isinstance(node, ast.Str):
                return node.s
            else:
                raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

        # Parse the arguments string as a function call
        tree = ast.parse(f"func({args_str})", mode='eval')
        func_call = tree.body  # ast.Call

        resolved_args = [resolve(arg) for arg in func_call.args]
        return tuple(resolved_args)

if __name__ == "__main__":
    args = parse_args()
    split = args.split
    output_path = args.output_path or os.path.join(REPO_ROOT, 'output', f'{split}.json')
    convs_output_path = convs_output_path_for(output_path)
    json_path = args.json_path or os.path.join(REPO_ROOT, 'data', split, f'rephrased_{split}.json')
    image_dir = args.image_dir or os.path.join(REPO_ROOT, 'data', split, 'images')

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Input JSON not found: {json_path} (pass --json_path to override)"
        )
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir} (pass --image_dir to override)"
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Loading LLM...")

    default_model_path = os.path.join(REPO_ROOT, "Qwen2.5-7B-Instruct")
    model_name_or_path = args.model or default_model_path

    llm = load_hf_chat_llm(
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

    print(f"Model loaded: {model_name_or_path} (quantization={args.quantization})")

    error_budget = 1

    prev_results_path = output_path
    results = []
    convs = []
    answered_ids = set()

    print('Split:', split)
    print('Input JSON:', json_path)
    print('Image dir:', image_dir)
    print('Saving results to:', output_path)
    print('Saving conversations to:', convs_output_path)

    if prev_results_path and os.path.exists(prev_results_path):
        with open(prev_results_path, 'r') as f:
            prev_results = json.load(f)
            results = [item for item in prev_results if item['normalized_answer'] != "-1"]
        if os.path.exists(convs_output_path):
            with open(convs_output_path, 'r') as f:
                prev_convs = json.load(f)
                convs = [item for item in prev_convs if item['normalized_answer'] != "-1"]
        answered_ids = {item['id'] for item in results}
        print(f"Loaded {len(results)} previous results.")

    tools = tools_api(dist_model_cfg={'model_path': os.path.join(REPO_ROOT, 'distance_est', 'ckpt', 'epoch_5_iter_6831.pth')}, 
                      inside_model_cfg={'model_path': os.path.join(REPO_ROOT, 'inside_pred', 'ckpt', 'epoch_4.pth')},
                      small_dist_model_cfg={'model_path': os.path.join(REPO_ROOT, 'distance_est', 'ckpt', '3m_epoch6.pth')},
                      resize=(360, 640),
                      mask_IoU_thres=0.3, inside_thres=0.5,
                      cascade_dist_thres=300, clamp_distance_thres=25)

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for idx, item in tqdm(enumerate(data), total=len(data)):
        id = item['id']
        if id in answered_ids:
            continue
            
        agent = Agent(
            llm,
            tools,
            item,
            image_dir=image_dir,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        agent.set_masks()
        
        attempt = 0
        while attempt < error_budget:
            try:
                answer = agent.set_question()
                answer = agent.format_answer()

                if isinstance(answer, str) and answer.lower() in ['yes', 'no', 'true', 'false']:
                    print(f"Invalid answer format: {answer}.")
                    answer = "-1"
                
                results.append({
                    'id': id,
                    'normalized_answer': str(answer)
                })
                convs.append({
                    'id': id,
                    'normalized_answer': str(answer),
                    'conversation': agent.messages
                })

                break
            except Exception as e:
                attempt += 1
                print(f"Error processing item {id} (attempt {attempt}/{error_budget}): {e}")
                if attempt == error_budget:
                    results.append({
                        'id': id,
                        'normalized_answer': "-1"
                    })
                    convs.append({
                    'id': id,
                    'normalized_answer': "-1",
                    'conversation': agent.messages
                    })
    
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        with open(convs_output_path, 'w') as f:
            json.dump(convs, f, indent=4)
