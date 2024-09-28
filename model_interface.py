import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from litellm import completion, ModelResponse
import os
import pandas as pd
import pathlib
from typing import Dict, Optional, List

ROOT_DIR = pathlib.Path(__file__).parent
CACHE_DIR = ROOT_DIR / 'cache'
with open(ROOT_DIR / 'openrouter_api_key', 'r') as f:
    os.environ["OPENROUTER_API_KEY"] = f.read()


@dataclass
class CustomResponse:
    model_content: str
    n_input_tokens: int
    n_output_tokens: int
    used_cache: bool
    user_interaction_id: int


def cache_load(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Optional[Dict]:
    user_message = [m for m in messages if m['role'] == 'user'][0]['content']
    user_message_hash = hashlib.sha256(user_message.encode()).hexdigest()
    system_messages = [m for m in messages if m['role'] == 'system']
    if system_messages:
        system_message_hash = hashlib.sha256(system_messages[0]['content'].encode()).hexdigest()
    else:
        system_message_hash = ''
    interactions = pd.read_csv(CACHE_DIR / 'model_interaction_history.csv', index_col=False)
    interactions['system_hash'] = interactions['system_hash'].fillna(value='')
    existing_row = interactions.loc[
        (interactions['prompt_hash'] == user_message_hash)
        & (interactions['system_hash'] == system_message_hash)
        & (interactions['model'] == model)
        & (interactions['temperature'] == temperature)
    ]
    if not existing_row.empty:
        return {
            'model_content': existing_row['model_output'].iloc[0],
            'n_input_tokens': int(existing_row['n_input_tokens'].iloc[0]),
            'n_output_tokens': int(existing_row['n_output_tokens'].iloc[0])
        }
    return None


def cache_write(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        response: ModelResponse | Dict,
        used_cache: bool
    ) -> int:
    user_message = [m for m in messages if m['role'] == 'user'][0]['content']
    user_message_hash = hashlib.sha256(user_message.encode()).hexdigest()
    system_messages = [m for m in messages if m['role'] == 'system']
    if system_messages:
        system_message_hash = hashlib.sha256(system_messages[0]['content'].encode()).hexdigest()
    else:
        system_message_hash = ''
    if isinstance(response, ModelResponse):
        model_content = response.choices[0].message.content
        n_input_tokens = response.usage.prompt_tokens
        n_output_tokens = response.usage.completion_tokens
    elif isinstance(response, dict):
        model_content = response['model_content']
        n_input_tokens = response['n_input_tokens']
        n_output_tokens = response['n_output_tokens']
    else:
        raise TypeError(f"response is of type {type(response)}, expected dict or ModelResponse")

    # Write a model interaction
    interactions = pd.read_csv(CACHE_DIR / 'model_interaction_history.csv', index_col=False)
    interactions['system_hash'] = interactions['system_hash'].fillna(value='')
    existing_row = interactions.loc[
        (interactions['prompt_hash'] == user_message_hash)
        & (interactions['system_hash'] == system_message_hash)
        & (interactions['model'] == model)
        & (interactions['temperature'] == temperature)
        & (interactions['model_output'] == model_content)
    ]
    if existing_row.empty:
        new_model_interaction_id = interactions.shape[0]
        new_row = pd.DataFrame({
            'id': [new_model_interaction_id],
            'prompt_hash': [user_message_hash],
            'system_hash': [system_message_hash],
            'model': [model],
            'temperature': [temperature],
            'model_output': [model_content],
            'n_input_tokens': [n_input_tokens],
            'n_output_tokens': [n_output_tokens],
        })
        interactions = pd.concat([interactions, new_row]) if (not interactions.empty) else new_row
        interactions.to_csv(CACHE_DIR / 'model_interaction_history.csv', index=False)
    else:
        new_model_interaction_id = existing_row['id'].iloc[0]

    # Write the prompt-hash pair
    prompt_hash_file_path = CACHE_DIR / 'hash_to_prompt' / f'{user_message_hash}.txt'
    if not prompt_hash_file_path.is_file():
        with open(prompt_hash_file_path, 'w+') as f:
            f.write(user_message)
    if system_message_hash != '':
        prompt_hash_file_path = CACHE_DIR / 'hash_to_prompt' / f'{system_message_hash}.txt'
        if not prompt_hash_file_path.is_file():
            with open(prompt_hash_file_path, 'w+') as f:
                f.write(system_messages[0]['content'])

    # Write a user interaction
    with open(CACHE_DIR / 'user_interaction_history.csv', 'r') as f:
        reader = csv.DictReader(f)
        history_lines = list(reader)
    latest_user_interaction_id = int(history_lines[-1]['id']) if history_lines else 0
    new_user_interaction_id = latest_user_interaction_id + 1
    new_row = {
        'id': new_user_interaction_id,
        'model_interaction_id': new_model_interaction_id,
        'timestamp': datetime.utcnow().isoformat(timespec='milliseconds'),
        'used_cache': str(used_cache)
    }
    with open(CACHE_DIR / 'user_interaction_history.csv', 'a', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=new_row.keys())
        writer.writerow(new_row)
    return latest_user_interaction_id


def prompt_model(
        messages: List[Dict[str, str]],
        model: str = 'openrouter/meta-llama/llama-3.1-8b-instruct:free',
        temperature: float = 0.3,
        write_to_cache: bool = True,
        load_from_cache: bool = True
    ) -> CustomResponse:
    if load_from_cache:
        response = cache_load(model=model, messages=messages, temperature=temperature)
        used_cache = True
    if (not load_from_cache) or (response is None):
        used_cache = False
        response = completion(model=model, messages=messages, temperature=temperature)
    if write_to_cache:
        user_interaction_id = cache_write(model=model, messages=messages, temperature=temperature, response=response,
                                          used_cache=used_cache)
    else:
        user_interaction_id = None
    if isinstance(response, ModelResponse):
        custom_response = CustomResponse(
            model_content=response.choices[0].message.content,
            n_input_tokens=response.usage.prompt_tokens,
            n_output_tokens = response.usage.completion_tokens,
            used_cache=used_cache,
            user_interaction_id=user_interaction_id
        )
    elif isinstance(response, dict):
        custom_response = CustomResponse(
            model_content=response['model_content'],
            n_input_tokens=response['n_input_tokens'],
            n_output_tokens=response['n_output_tokens'],
            used_cache=used_cache,
            user_interaction_id=user_interaction_id
        )
    else:
        raise TypeError(f"response is of type {type(response)}, expected dict or ModelResponse")
    return custom_response


if __name__ == '__main__':
    prompt_model([{'role': 'system', 'content': 'you are buzz lightspeed'}, {'role': 'user', 'content': 'hello'}])