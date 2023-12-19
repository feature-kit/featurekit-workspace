from threading import Thread
from queue import Queue
import time
from openai import OpenAI
import os
from tqdm import tqdm

client = OpenAI()


def threaded_map(fn, input, n_threads=20):
    input_queue = Queue()
    for idx, item in enumerate(input):
        input_queue.put((idx, item))
    results_queue = Queue()

    def worker():
        while not input_queue.empty():
            idx, item = input_queue.get()
            result = fn(item)
            results_queue.put((idx, result))
            input_queue.task_done()
            # time.sleep(1)
    
    for _ in range(n_threads):
        thread = Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    input_queue.join()
    
    results = []
    while not results_queue.empty():        
        results.append(results_queue.get())

    return [result for idx, result in sorted(results, key=lambda x: x[0])]



def get_chat_prompt(prompt_info, messages, post_prompt=False):
    if isinstance(prompt_info, str):
        prompt_messages = [{'role': 'system', 'content': prompt_info}, {'role': 'user', 'content': messages}]
    elif isinstance(prompt_info, list):
        prompt_messages = prompt_info
    else:
        raise TypeError(f"prompt_info must be str or list[dict[str, str]], not {type(prompt_info)}")

    if post_prompt is not False:
        prompt_messages += [{'role': 'system', 'content': post_prompt}]
    
    return prompt_messages

def get_base_prompt(prompt_info, messages, post_prompt=False):
    if isinstance(prompt_info, str):
        prompt = prompt_info
    elif isinstance(prompt_info, list):
        prompt = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in prompt_info])
    else:
        raise TypeError(f"prompt_info must be str or list[dict[str, str]], not {type(prompt_info)}")

    if post_prompt is not False:
        if isinstance(post_prompt, str):
            prompt += "\n" + post_prompt
        elif isinstance(post_prompt, list):
            prompt += "\n" + "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in post_prompt])
        else:
            raise TypeError(f"post_prompt must be str or list[dict[str, str]], not {type(post_prompt)}")
    
    return prompt


def format_messages(messages, *format_args, **format_kwargs):
   return [{'role': message['role'], 'content': message['content'].format(*format_args, **format_kwargs)} for message in messages]
    
def format_prompt(prompt, *format_args, **format_kwargs):
    if isinstance(prompt, str):
        return prompt.format(*format_args, **format_kwargs)
    elif isinstance(prompt, list):
        return format_messages(prompt, *format_args, **format_kwargs)

def reverse_roles(messages):
    return [{'role': 'user' if message['role'] == 'assistant' else 'assistant', 'content': message['content']} for message in messages]



def query(prompt, messages, model, reversed_roles = False, max_tokens=300, **kwargs):
        '''
        prompt: list[dict[str, str]]
        reversed_roles:
            True when the model is acting as the user and False when the model is acting as the assistant
            When reversed_roles=True, chat prompts must treat the chat model as the 'assistant', and not as the 'user'.
              Text prompts refer to the text model as the 'user' and the chat model as the 'assistant'.
              It's supposed to be interpretd as 'are they effectively the user in the conversation'.

        returns a single string, the response from model
        '''

        if isinstance(prompt, str):
            prompt = [{'role': 'system', 'content': prompt}]
        
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]

        if reversed_roles:
            messages = reverse_roles(messages)

        for retry_attempt in range(1, 6):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages = prompt+messages,
                    max_tokens=max_tokens,
                    **kwargs
                )
                break
            except Exception as e:
                print(e)
                print('Model: ', model)
                print('kwargs: ', kwargs)
                print('prompt: ', str(prompt).replace('\n', '\\n'))

                if retry_attempt == 5:
                    raise e
                retry_interval = int(15*(retry_attempt**1.6))
                print(f'Sleeping for {retry_interval} seconds...')
                time.sleep(retry_interval)
                continue
        return response#response['choices'][0]['message']['content'].strip()



def threaded_map(fn, inputs, n_threads=20, args=[], kwargs={}, input_is_kwargs=False):
    pbar = tqdm(total=len(inputs))
    input_queue = Queue()
    for idx, item in enumerate(inputs):
        input_queue.put((idx, item))
    results_queue = Queue()

    def worker():
        while not input_queue.empty():
            idx, item = input_queue.get()
            if input_is_kwargs is True:
                print('tick')
                result = fn(*args, **item, **kwargs)
            else:
                result = fn(item, *args, **kwargs)
            results_queue.put((idx, result))
            input_queue.task_done()
            pbar.update(1)
            # time.sleep(1)
    
    for _ in range(n_threads):
        thread = Thread(target=worker)
        thread.daemon = True
        thread.start()
    
    input_queue.join()
    
    results = []
    while not results_queue.empty():        
        results.append(results_queue.get())

    return [result for idx, result in sorted(results, key=lambda x: x[0])]


def interruptible_threaded_map(fn, inputs, n_threads=20, args=[], kwargs={}):
    kill_threads_now = False

    def interrupt_map():
        nonlocal kill_threads_now
        kill_threads_now = True

    def run_threaded_map():
        nonlocal kill_threads_now
        pbar = tqdm(total=len(inputs))
        input_queue = Queue()
        for idx, item in enumerate(inputs):
            input_queue.put((idx, item))
        results_queue = Queue()

        def worker():
            while not input_queue.empty():
                idx, item = input_queue.get()
                result = fn(item, *args, **kwargs)
                results_queue.put((idx, result))
                input_queue.task_done()
                pbar.update(1)
                
                if kill_threads_now:
                    print('killing thread..')
                    break
        
        for _ in range(n_threads):
            thread = Thread(target=worker)
            thread.daemon = True
            thread.start()
        
        input_queue.join()

        if kill_threads_now:
            return None
        
        results = []
        while not results_queue.empty():        
            results.append(results_queue.get())

        return [result for idx, result in sorted(results, key=lambda x: x[0])]

    return run_threaded_map, interrupt_map

def threaded_query(prompt, inputs, model="gpt-3.5-turbo", n_threads=10, *args, **kwargs):
    '''
    template: str or list[dict[str, str]] or None
      depending on if model is LM or chat model, template is either a string or a list of dicts
    model: str
    n_threads: int
    template_args: list
      list of lists of args to pass to template per model query
    template_kwargs: list
      list of dicts of kwargs to pass to template per model query
    '''

    if isinstance(prompt, str):
        prompt = [{'role': 'system', 'content': prompt}]

    n_threads = max(n_threads, len(inputs))

    def query_helper(messages):
        return query(prompt, messages, model=model, *args, **kwargs)

    return threaded_map(query_helper, inputs, n_threads=n_threads)