import random
from datasets import load_dataset
import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import openai
import os
import logging
from hydra.utils import get_original_cwd, to_absolute_path
from tqdm import tqdm


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:

    openai.api_key = os.getenv("OPENAI_API_KEY")

    fname = os.path.join(get_original_cwd(), config['output_dir'], 'log.txt')
    wfile = open(fname, 'w')

    # data loader for ml100k
    item = load_dataset('datasets/ml100k.py', 'item', split='data')
    # create iid to title dict
    iid_dict = {i['iid']: i['title'] for i in item}
    # load ml100k seq dataset
    seq = load_dataset('datasets/ml100k_seq.py', split='test')

    random.seed(config['seed'])

    # Preprocessing the datasets
    def preprocess_function(examples):
        # Tokenize the texts
        candidates_and_answer = examples['candidates'] + [examples['target']]
        random.shuffle(candidates_and_answer)
        examples["can_input"] = '\n'.join([f'{i+1}. {iid_dict[iid]}' for i, iid in enumerate(candidates_and_answer)])
        examples["input"] = '\n'.join([f'{iid_dict[iid]}' for iid in examples['seq']])
        # examples["input"] = '; '.join([iid_dict[iid] for iid in examples['seq']])
        examples["answer"] = iid_dict[examples['target']]
        return examples

    processed_datasets = seq.map(
        preprocess_function,
        # remove_columns=seq.column_names,
        desc="To Natural language and Applying prompts",
    )

    ndcg = evaluate.load('metric/ndcg.py', experiment_id='ndcg')
    ndcg_random = evaluate.load('metric/ndcg.py', experiment_id='ndcg_random')
    error = []
    for cnt, d in tqdm(enumerate(processed_datasets)):
        if cnt == config['max_test_samples']:
            break
        prompt = config['template']['prompt']
        prompt = prompt.replace("[SEQ]", d['input'])
        prompt = prompt.replace("[CANDIDATES]", d['can_input'])
        # print(prompt)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": config['template']['system']},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        try:
            text_results = completion.choices[0].message["content"]
            print(text_results)
            split_results = text_results.split(',')
            print(split_results)
            split_results = [s.strip() for s in split_results]
            print(split_results)
            if len(split_results) != 11:
                error[cnt] = text_results
                print(len(split_results))
        except:
            error[cnt] = text_results
            continue
        
        prediction = [1 if j == d['target'] else 0 for j in split_results]
        ndcg.add(prediction=prediction)
        break
        # print(completion.choices[0].message)
        # break
        print(prompt, file=wfile)
        print(text_results, file=wfile)
        
        ### for random baseline ###
        candidates_and_answer = d['candidates'] + [d['target']]
        random.shuffle(candidates_and_answer)
        prediction = [1 if j == d['target'] else 0 for j in candidates_and_answer]
        ndcg_random.add(prediction=prediction)


    # ndcg_results = ndcg.compute(k=[10, 20, 50, 100])
    ndcg_results = ndcg.compute(k=10)
    # ndcg_results = ndcg.compute()

    ndcg_random_results = ndcg_random.compute(k=10)

    log.info('NDCG results')
    for k, v in ndcg_results.items():
        log.info(f'{k}: {v}')
    
    log.info('NDCG random results')
    for k, v in ndcg_results.items():
        log.info(f'{k}: {v}')

    if len(error) > 0:
        log.error('Error cases')
        for k, v in error.items():
            log.error(f'{k}: {v}')

if __name__ == '__main__':
    main()
