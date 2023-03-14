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
from time import sleep


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:

    openai.api_key = os.getenv("OPENAI_API_KEY")

    fname = os.path.join(get_original_cwd(), config['output_dir'], 'log.txt')
    wfile = open(fname, 'w')

    # data loader for ml100k
    item = load_dataset('datasets/ml100k.py', 'item', split='data')
    data = load_dataset('datasets/ml100k.py', 'data', split='data')

    # create iid to title dict
    iid_dict = {i['iid']: i['title'] for i in item}
    # create iid to popularity dict
    iid_pop_dict = {i['iid']: 0 for i in item}
    for d in data:
        iid_pop_dict[d['iid']] += 1

    # load ml100k seq dataset
    seq = load_dataset('datasets/ml100k_seq.py', split='test')

    random.seed(config['seed'])

    # Preprocessing the datasets
    def preprocess_function(examples):
        # Tokenize the texts
        candidates_and_answer = examples['candidates'] + [examples['target']]
        random.shuffle(candidates_and_answer)
        examples["candidates_and_answer"] = candidates_and_answer
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

    error = {}
    for cnt, d in tqdm(enumerate(processed_datasets)):
        if cnt == config['max_test_samples']:
            break
        prompt = config['template']['prompt']
        prompt = prompt.replace("[SEQ]", d['input'])
        prompt = prompt.replace("[CANDIDATES]", d['can_input'])
    
        print(cnt, prompt, file=wfile)

        ### openai chatGPT ###
        if config['method'] == 'openai':
            # Try 4 times for successful request
            error_flag = False
            for x in range(0, 4):  
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages=[
                            {"role": "system", "content": config['template']['system']},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0
                    )
                    break
                except:
                    error_flag = True
                    sleep(2)
                    
            if error_flag:
                error[cnt] = "Max retries exceeded with url"
                continue

            # Check for invalid generation
            try:
                text_results = completion.choices[0].message["content"]
                print(text_results, file=wfile)
                print(file=wfile)
                split_results = text_results.split(',')
                split_results = [int(s.strip()) for s in split_results]
                if len(split_results) != 11:
                    error[cnt] = text_results
                    continue
            except:
                error[cnt] = text_results
                continue
            
            ordered_list = [d['candidates_and_answer'][i-1] for i in split_results]

        ### popularity baseline ###
        elif config['method'] == 'popularity':
            candidate_pop_dict = {j: iid_pop_dict[j] for j in d['candidates_and_answer']}
            sorted_candidate_pop_dict = sorted(candidate_pop_dict.items(), key=lambda x: x[1], reverse=True)
            ordered_list = [j[0] for j in sorted_candidate_pop_dict]
        ### random baseline ###
        else:
            ordered_list = d['candidates_and_answer']
            random.shuffle(ordered_list)

        prediction = [1 if j == d['target'] else 0 for j in ordered_list]
        ndcg.add(prediction=prediction)


    # ndcg_results = ndcg.compute(k=[10, 20, 50, 100])
    ndcg_results = ndcg.compute(k=10)
    # ndcg_results = ndcg.compute()

    log.info(f'NDCG results for {config["method"]} method')
    for k, v in ndcg_results.items():
        log.info(f'{k}: {v}')

    if len(error) > 0:
        log.info('Error cases')
        for k, v in error.items():
            log.info(f'{k}: {v}')

if __name__ == '__main__':
    main()
