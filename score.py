import numpy as np
from datasets import load_metric
import config

metric = load_metric('glue', config.task)

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))



def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if config.task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

