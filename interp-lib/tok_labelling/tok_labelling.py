
import torch
from openai import OpenAI
from tokenizers import Tokenizer
import json
from noa_tools import threaded_map, query
from tqdm import tqdm
import random
import itertools
import os

def local_fname(fname):
    return os.path.join(os.path.dirname(__file__), fname)



client = OpenAI()


top_10k = torch.load(local_fname('top_10k_indices.pt'))
tokenizer = Tokenizer.from_pretrained('roneneldan/TinyStories-33M')

toks = [tokenizer.decode([tok_id]) for tok_id in top_10k]
toks = list(set([tok.lower()[1:] for tok in toks if tok[0] == ' ']))


def format_nest(nest, **kwargs):
    if isinstance(nest, list):
        return [format_nest(item, **kwargs) for item in nest]
    elif isinstance(nest, dict):
        return {format_nest(k, **kwargs): format_nest(v, **kwargs) for k, v in nest.items()}
    elif isinstance(nest, str):
        return nest.format(**kwargs)

prompt_template = '''Given the list of strings below, identify which strings match the following description:
{description}
To log the strings that match the description, call log_{title}. Copy strings exactly as they appear.

Here is the list of strings for you to process: {batch}'''

functions_template = [
    {
    "name": "log_{title}",
    "description": "",
    "parameters": {
        "type": "object",
        "properties": {
            "identified_{title}": {
                "type": "array",
                "description": "{description}",
                "items": {
                    "type": "string",
                }
            },
        },
    },
    "required": ["log_{title}"],
    },
]

def get_tok_strs_from_batch(batch, title, description, model='gpt-4',):
    out = query(prompt=prompt_template.format(title=title, description=description, batch=batch),
                messages=[],
                model=model,
                functions=format_nest(functions_template, title=title, description=description),
                function_call={'name': f"log_{title}"},
                temperature=0.9)

    try:
        outputted_words = json.loads(
            out.choices[0].message.function_call.arguments
        )[f"identified_{title}"]
    except Exception as e:
        print(f'Exception: {e}')
        return []

    words = []
    for candidate_token in outputted_words:
        if candidate_token not in batch:
            continue
        for tok in [' '+candidate_token, candidate_token.capitalize(), ' '+candidate_token.capitalize(), candidate_token]:
            ids = tokenizer.encode(tok).ids
            if len(ids) != 1:
                continue
            words.append(tok)

    return words


def new_tok_label(title, description, batch_size=50, model='gpt-4'):
    '''
    title: str, title of label
    description: str, describing all tokens that fit the label.
    
    EG
    new_tok_label(title='past_tense_verbs', description='Strings that are past-tense verbs.')
    '''
    assert '-' not in title, 'snake_case please'
    assert ' ' not in title, 'snake_case please'
    assert title.lower() == title, 'snake_case please'

    ls = os.listdir(local_fname(f'tok_labels'))
    if title+'.json' in ls:
        with open(local_fname(f'tok_labels/{title}.json'), 'r') as f:
            existing_desc = json.load(f)['description']
        if description == existing_desc:
            print(f'A tok label, "{title}", already exists with this description.')
            return 
        inp = input(f'{title}.json already found in tok_labels/ with description:\n{existing_desc}\nOverwrite this file?')
        if inp.strip().lower() not in {'y', 'yes'}:
            print('Aborting tok labelling.')
            return
        else:
            pass



    counts = torch.zeros(50258).to(torch.int)

    for _ in range(3):
        random.shuffle(toks)
        batches = [toks[i:i+batch_size] for i in range(0, len(toks), batch_size)]
        out = list(itertools.chain.from_iterable(threaded_map(get_tok_strs_from_batch, batches, n_threads=20, kwargs={'title': title, 'description': description, 'model': model})))

        for word in out:
            tok_id = tokenizer.encode(word).ids[0]
            counts[tok_id] += 1

    identified_words = (counts >= 2).int().nonzero().flatten()
    identified_words = [tokenizer.decode([word]) for word in identified_words]
    identified_words = list(set(identified_words))
    result = {
        'title': title,
        'description': description,
        'tokens': identified_words
    }
    with open(local_fname(f'tok_labels/{title}.json'), 'w') as f:
        json.dump(result, f)
    print(f'Saved {len(identified_words)} words in {title}.json!')

import os

def load_tok_label(title):
    available_labels = [item.split('.')[0] for item in os.listdir(local_fname('tok_labels')) if item.split('.')[1] == 'json']
    if title not in available_labels:
        print(f'tok label "{title}" not found. Available tok labels are: {available_labels}')
    else:
        with open(local_fname(f'tok_labels/{title}.json'), 'r') as f:
            res = json.load(f)
        description = res['description']
        labelled_toks = res['tokens']
        print(f'Loading {title}: {description}')
        return labelled_toks


