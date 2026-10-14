import ollama
import re
import json
import requests
import os
import cv2
import base64
from utils.vln_logger import vln_logger
from textblob import TextBlob
import numpy as np

def load_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data

def save_json(data, filename):
    """
    Saves a Python dictionary (or any serializable object) to a JSON file.

    Args:
    - data: The data to save (should be a dictionary or list).
    - filename: The file path (including filename) where the data should be saved.
    """
    try:
        # Open the file in write mode and dump the JSON data into it
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        # print(f"Data has been successfully saved to {filename}")
    
    except Exception as e:
        print(f"Error saving the data to {filename}: {e}")

def construct_query(phase, **placeholders):
    """
    Construct a natural language query for the LLM.
    
    Args:
        phase (dict): Dictionary containing the phase prompts.
        **placeholders: Arbitrary keyword arguments representing the dynamic placeholders.

    Returns:
        str: The formatted query string.
    """
    phase_prompt = "\n\n".join(phase['phase_prompt'])
    content = phase_prompt.format(**placeholders)
    return content

def construct_target_query(candidate_object, goal_class, image_history):
    '''
    Input:
        candidate_object: dict, store all candidate object info, from the object list
        goal_class: str, the goal class name
    Output:
        prompt: str, the prompt for the LLM
        image: np.ndarray, the image with xyxy bbox of the candidate object
    '''
    prompt = f"You are a verification agent that helps a robot in a 3D environment looking for objects in the scene. Please verify if the object in the image enclosed by the red bounding box is a '{goal_class}' or not. Answer with 'yes' or 'no' with no additional text." 
    
    # if the image with highest confidence is not the same as the goal class, then it is not a valid candidate
    max_conf_index = np.argmax(candidate_object['conf'])
    xyxy = candidate_object['xyxy'][max_conf_index]
    image_idx = candidate_object['image_idx'][max_conf_index]
    image = np.array(image_history[image_idx])
    # draw red bounding box on the image
    image = cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
    
    return prompt, image
    
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _build_messages(prompt, llm_name, image):
    if image is None:
        if llm_name == 'gpt':
            return [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        return [{'role': 'user', 'content': prompt}]
    if isinstance(image, str) and os.path.exists(image):
        base64_image = encode_image(image)
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.png', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    if 'gpt' in llm_name:
        return [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {
                'url': f'data:image/png;base64,{base64_image}',
            }},
        ]}]
    return [{'role': 'user', 'content': prompt, 'images': [base64_image]}]


def _query_gpt(llm_name, messages):
    max_tokens_map = {
        'gpt-4o': 16384,
        'gpt-4o-2024-11-20': 16384,
        'o1-preview': 32768,
        'gpt-4o-mini': 16384,
        'gpt-4.1-nano': 16384,
    }
    model = 'gpt-4o-mini' if llm_name == 'gpt' else llm_name
    api_key = os.environ.get('GPT_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('Set OPENAI_API_KEY before using a GPT model.')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    response = requests.post(
        base_url.rstrip('/') + '/chat/completions',
        headers={'Authorization': f'Bearer {api_key}'},
        json={'model': model, 'messages': messages,
              'max_completion_tokens': max_tokens_map[model]},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def query_llm(prompt, llm_name='gpt-4o-mini', image=None):
    if 'gpt' in llm_name:
        llm_label = 'gpt'
    elif 'llama' in llm_name:
        llm_label = 'llama'
    elif 'qwen' in llm_name:
        llm_label = 'qwen'
    else:
        llm_label = 'other'
    vln_logger.increase_function_call_count('query_llm', args={'llm_name': llm_label})
    messages = _build_messages(prompt, llm_name, image)
    if 'gpt' in llm_name:
        return _query_gpt(llm_name, messages)
    client = ollama.Client(host=os.getenv('OLLAMA_HOST'))
    return client.chat(model=llm_name, messages=messages).message.content.lower()


def format_graph(response):
    regex = r"(.+?)\n```.*?\n(.*?)```"
    matches = re.finditer(regex, response, re.DOTALL)
    graph = None
    for match in matches:
        graph = match.group(2)
        try:
            graph = json.loads(graph)
            break
        except:
            graph = None
            pass
    return graph

# def format_response(response):
#     response = response.lower().strip()

#     # Regex to match **answer:**, # " or a direct answer
#     match = re.search(r"\*\*answer:\*\* (.+)|#\s*answer:\s*(.+)|\*answer:\* (.+)|\n\nanswer:\n\n (.+)|# \"(.+)\"|# \s*(\w[\w\s]*)?|^\s*(\w[\w\s]*)$", response, re.MULTILINE)

#     if match:
#         answer = match.group(1) or match.group(2) or match.group(3) or match.group(4) or match.group(5) or match.group(6) or match.group(7)
#         answer = answer.strip()
#     else:
#         answer = response
#     return answer

def clean_markdown(text):
    return re.sub(r'^[#\*]+|[#\*]+$', '', text).strip()

def correct_text(text):
    return str(TextBlob(text).correct())

def format_response(response):
    try:
        response = response.lower().strip()

        # Updated pattern with more formats added
        pattern = re.compile(
            r"\*\*answer:\*\*\s*(.+)"            # **Answer:** value
            r"|\*answer:\*\s*(.+)"               # *Answer:* value
            r"|#\s*answer:\s*(.+)"               # # Answer: value
            r"|\n\nanswer:\n\n\s*(.+)"           # Answer in a block
            r"|>\s*answer:\s*(.+)"               # > Answer: value (blockquote style)
            r"|###\s*answer\s*[:\-]?\s*(.+)"     # ### Answer or ### Answer:
            r"|#\s*\"(.+?)\""                    # # "answer"
            r"|\*\*(.+?)\*\*"                    # **answer**
            r"|\*(.+?)\*"                        # *answer*
            r"|^\s*([\w\s\-]+)$",                # plain answer
            re.MULTILINE
        )

        match = pattern.search(response)

        if match:
            # Get the first non-None group
            answer = next(group for group in match.groups() if group)
            answer = clean_markdown(answer)
            answer = correct_text(answer)
        else:
            answer = clean_markdown(response)
            answer = correct_text(answer)

        return answer.strip()
    except:
        print("[format_response] ERROR")
        print(response)
        return ""


def text2value(text):
    try:
        value = float(text)
    except:
        value = 0
    return value
