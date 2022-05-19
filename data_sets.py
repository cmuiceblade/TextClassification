import datasets
import random
from datasets import load_dataset, load_from_disk
import pandas as pd
from IPython.display import display, HTML
import config


def load_cola_data():
    dataset = load_dataset('glue', config.task, cache_dir=config.data_path)
    dataset.save_to_disk('./coladataset')
    #dataset = load_from_disk(config.data_path)
    return dataset

def show_random_samples(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't sample more data than dataset holds."

    picks = []

    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    
    for column, type in dataset.features.items():
        if isinstance(type, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: type.names[i])
    
    display(HTML(df.to_html()))
        


ret = load_cola_data()
print(ret)
#show_random_samples(dataset["train"])