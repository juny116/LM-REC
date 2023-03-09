import random
from datasets import load_dataset
import evaluate
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:

    # data loader for ml100k
    item = load_dataset('datasets/ml100k.py', 'item', split='data')
    # create iid to title dict
    iid_dict = {i['iid']: i['title'] for i in item}
    # load ml100k seq dataset
    seq = load_dataset('datasets/ml100k_seq.py', split='test')

    prompt = config['template']['prompt']

    # Preprocessing the datasets
    def preprocess_function(examples):
        # Tokenize the texts
        examples["input"] = prompt + '; '.join([iid_dict[iid] for iid in examples['seq']])
        examples["answer"] = iid_dict[examples['target']]
        return examples

    processed_datasets = seq.map(
        preprocess_function,
        # remove_columns=seq.column_names,
        desc="To Natural language and Applying prompts",
    )

    ndcg = evaluate.load('metric/ndcg.py', experiment_id='ndcg')
    random.seed(config['seed'])

    # print first sample input
    print(processed_datasets[0]['input'])

    # test ndcg for the first 20 samples, use shuffled list as dummy predictions
    for i in range(20):
        candidates_and_answer = processed_datasets[i]['candidates'] + [processed_datasets[i]['target']]
        random.shuffle(candidates_and_answer)
        # print('Prediction: ', candidates_and_answer)
        # print('Target: ', processed_datasets[i]['target'])

        prediction = [1 if j == processed_datasets[i]['target'] else 0 for j in candidates_and_answer]
        # print(prediction)

        ndcg.add(prediction=prediction)

    ndcg_results = ndcg.compute(k=[10, 20, 50, 100])
    # ndcg_results = ndcg.compute()
    print(ndcg_results)

if __name__ == '__main__':
    main()
