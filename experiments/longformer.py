import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.system("pip3 install transformers datasets rouge-score nltk -q")
import nltk
import numpy as np 
import pandas as pd
import torch
from datasets import Dataset
from datasets import load_metric
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, help='File path containing files for train')
parser.add_argument('--dev_path', type=str, help='File path containing files for validation')
parser.add_argument('--test_path', type=str, help='File path containing files for test')
parser.add_argument('--train_oracle_paragraph_path', type=str, help='File path containing oracle paragraphs files for train')
parser.add_argument('--dev_oracle_paragraph_path', type=str, help='File path containing oracle paragraphs files for validation')
parser.add_argument('--test_oracle_paragraph_path', type=str, help='File path containing oracle paragraphs files for test')
parser.add_argument('--filtered_source', type=str, default='none', help='Whether to use filtered source judgement, values can be none | acts | paras')

parser.add_argument('--save_at_end', action='store_const', const=True, help='save model after training', default=True)
parser.add_argument('--output_directory', type=str)

parser.add_argument('--experiment_name', type=str, default='legal')
parser.add_argument('--test', action='store_const', const=True, default=False)

parser.add_argument('--continue_training', type=str)
parser.add_argument('--load_from_checkpoint', type=str, help='initialize model from checkpoint')
parser.add_argument('--tokenizer_path', type=str, help='path to tokenizer', default='xlm-roberta-base')
parser.add_argument('--seed', type=int, default=2024)

parser.add_argument('--max_seq_len', type=int, default=256)
parser.add_argument('--max_input_length', type=int, default=4096)
parser.add_argument('--truncate_left', action='store_const', const=True, default=False)
parser.add_argument('--source_prefix', type=str)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int)
parser.add_argument('--num_gpus', type=int)
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=2500)
parser.add_argument('--eval_steps', type=int, default=2500)
parser.add_argument('--max_steps', type=int, default=8000)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--full_size_corpus', action='store_const', const=True, default=False)
parser.add_argument('--no_optimizer_scheduler', action='store_const', const=True, default=False)
parser.add_argument('--no_save_every_epoch', action='store_const', const=True, default=False)
parser.add_argument('--no_early_stop', action='store_const', const=True, default=False)
parser.add_argument('--fp16', action='store_const', const=True, default=False)
parser.add_argument('--save_total_limit', type=int, default=999, help='The limit of the total amount of checkpoints. Will deletes the older checkpoints.')
parser.add_argument('--best_model_checkpoint_none', action='store_const', const=True, default=False)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--warmup_steps', type=int, default=-1)
args = parser.parse_args()
set_random_seeds(args.seed)

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}".format(torch.cuda.get_device_name(0)))
print('There are %d GPU(s) available.' % torch.cuda.device_count())



# train df
# training data in the form of csv file
training_df = pd.read_csv(args.train_path)

if args.filtered_source == 'none':
    training_article_ls = list(training_df['text'])
elif args.filtered_source == 'paras':
    training_oracle_paragraph_df = pd.read_csv(args.train_oracle_paragraph_path)
    training_article_ls = list(training_oracle_paragraph_df['oracle_paragraphs'])
training_highlight_ls = list(training_df['summary'])
train = pd.DataFrame(columns=['target_text','source_text'])
train['target_text'] = training_highlight_ls
train['source_text'] = [str(item) for item in training_article_ls]
train = train.rename(columns={'target_text':'summary', 'source_text':'document'})
train = train.dropna()
train['document']= train['document'].apply(lambda x: x.lower())
train['summary'] = train['summary'].apply(lambda x: x.lower())
train = train.sample(frac=1).reset_index(drop=True)

# validation df
# validation data in the form of csv file
validation_df = pd.read_csv(args.dev_path)

if args.filtered_source == 'none':
    validation_article_ls = list(validation_df['text'])
elif args.filtered_source == 'paras':
    validation_oracle_paragraph_df = pd.read_csv(args.dev_oracle_paragraph_path)
    validation_article_ls = list(validation_oracle_paragraph_df['oracle_paragraphs'])
validation_highlight_ls = list(validation_df['summary'])
valid = pd.DataFrame(columns=['target_text','source_text'])
valid['target_text'] = validation_highlight_ls
valid['source_text'] = [str(item) for item in validation_article_ls]
valid = valid.rename(columns={'target_text':'summary', 'source_text':'document'})
valid = valid.dropna()
valid['document']= valid['document'].apply(lambda x: x.lower())
valid['summary'] = valid['summary'].apply(lambda x: x.lower())
valid = valid.sample(frac=1).reset_index(drop=True)

# test df
# validation data in the form of csv file
test_df = pd.read_csv(args.test_path)

if args.filtered_source == 'none':
    test_article_ls = list(test_df['text'])
elif args.filtered_source == 'paras':
    test_oracle_paragraph_df = pd.read_csv(args.test_oracle_paragraph_path)
    test_article_ls = list(test_oracle_paragraph_df['oracle_paragraphs'])
test_highlight_ls = list(test_df['summary'])
test = pd.DataFrame(columns=['target_text','source_text'])
test['target_text'] = test_highlight_ls
test['source_text'] = [str(item) for item in test_article_ls]
test = test.rename(columns={'target_text':'summary', 'source_text':'document'})
test = test.dropna()
test['document']= test['document'].apply(lambda x: x.lower())
test['summary'] = test['summary'].apply(lambda x: x.lower())

# model

model_checkpoint ='allenai/led-base-16384'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if args.truncate_left:
    assert tokenizer.truncation_side == 'right'
    tokenizer.truncation_side = 'left'
    assert tokenizer.truncation_side == 'left'

max_input_length = args.max_input_length
max_target_length = 256
batch_size = args.batch_size

def preprocess_function(examples):
    prefix = args.source_prefix if args.source_prefix is not None else ""
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding='max_length')

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train = Dataset.from_pandas(train)
valid = Dataset.from_pandas(valid)

tokenized_train = train.map(preprocess_function, batched=True)
tokenized_valid = valid.map(preprocess_function, batched=True)

# model instance
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# set generate hyperparameters
model.config.num_beams = 3
model.config.max_length = 256
model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# metrics

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

model_name = model_checkpoint.split("/")[-1]
seq2seq_args = Seq2SeqTrainingArguments(
    f"out/{model_name}-finetuned-{args.experiment_name}",
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=args.epochs,
    disable_tqdm=False,
    max_steps=args.max_steps,
    logging_steps=500,  
    save_steps=500,  
    eval_steps=500,  
    warmup_steps=500,  
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    local_rank=int(os.environ['LOCAL_RANK']),
    fp16=args.fp16,
    gradient_accumulation_steps=args.gradient_accumulation_steps
)

trainer = Seq2SeqTrainer(
    model,
    seq2seq_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

metric = load_metric("rouge")

# training

trainer.train()

# save best checkpoint
trainer.save_model(output_dir=f"out/{model_name}-finetuned-{args.experiment_name}/best/")

# inference 

eval_dataset = Dataset.from_pandas(test)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
predict_results = trainer.predict(eval_dataset, max_length=max_target_length, num_beams=3)
metrics = predict_results.metrics
print(metrics)

predictions = tokenizer.batch_decode(
    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
predictions = [pred.strip() for pred in predictions]
output_prediction_file = os.path.join(f"out/{model_name}-finetuned-{args.experiment_name}/best/", "test_predictions.csv")
df = pd.DataFrame(predictions, columns=["summary"])
df.to_csv(output_prediction_file, index=False)


