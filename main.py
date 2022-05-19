from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import config 
import data_sets
import score

tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


sentence1_key, sentence2_key = task_to_keys[config.task]


def preprocess(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], trancation=True)

dataset = data_sets.load_cola_data()

# ret = preprocess(dataset['train'][:5])
# print(ret)

encoded_dataset = dataset.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(config.model_checkpoint, num_labels=config.num_labels)

args = TrainingArguments(
    "test_glue",
    evaluation_strategy = "epoch",
    save_strategy = 'epoch',
    learning_rate = config.learning_rate,
    per_device_train_batch_size = config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=config.metric_name,
    output_dir=config.output_dir
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[config.validation_key],
    tokenizer=tokenizer,
    compute_metrics=score.compute_metrics
)

trainer.train()

trainer.evaluate()



