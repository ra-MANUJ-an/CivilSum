# Summarization Experiments
Here you can find the code and scripts to run experiments with the extractive baselines and the abstractive models (FactorSum, Longformer, Llama-2) mentioned in the paper. 

The eperiments use `python 3.8.8` with dependencies specified in the [requirements.txt](./requirements.txt) file. Also, make sure you download the following data files to a `data` folder:

- Documents and summaries:
  - [train.csv](https://drive.google.com/file/d/1WX5w-ClK82Yy916350bRp3hcuqab1D-x/view?usp=sharing)
  - [validation.csv](https://drive.google.com/file/d/1khcDFSca6zUv-dFQ4XaxMt8jRf-9PZbV/view?usp=sharing)
  - [test.csv](https://drive.google.com/file/d/1v0EM35CFWxFPc5D4bFxpf59mV_kEv0pw/view?usp=sharing)

- Pre-processed document paragraphs:
  - [paragraphs_train.csv:](https://drive.google.com/file/d/12tdNRnnbovRA1E-YItkKxzEoXfunctBd/view?usp=sharing)
  - [paragraphs_valid.csv](https://drive.google.com/file/d/1eyt6ekCQuIQw7vs4cscwKgxLyqbmVPvN/view?usp=sharing)
  - [paragraphs_test.csv](https://drive.google.com/file/d/1sa6xT14SiZRm7-X0cG0XK6GXhq_P8FrR/view?usp=sharing)

To independently find the pre-processed document paragraphs, please consult the `process_data` function provided in the `util.py` file. Below is an example of its usage:

```python
import pandas as pd
from util import find_paragraph_refs, process_data

train = pd.read_csv("train.csv")
train_paragraphs = find_paragraph_refs(train.summary)
processed_data = process_data(train, train_paragraphs)
pd.to_csv("paragraphs_train.csv", index=False)
```

## Extractive baselines
For random extractive baseline:
```
python baselines.py --data_path /path/to/data/test.csv --source text --random
```

For the "paragraphs" baseline:
```
python baselines.py --data_path /path/to/data/test.csv --source oracle_paragraphs
```

## FactorSum
First, clone the FactorSum repository:
```
git clone https://github.com/thefonseca/factorsum.git
```

Before inference or training, preprocess the dataset splits:
```
python experiments/fsum.py preprocess /path/to/data/folder/
```

To generate summaries using content guidance from "oracle paragraphs", use the following command:
```
python experiments/fsum.py evaluate \
--data_dir /path/to/data/folder/ \
--content_type oracle \
--output_dir output
```

In this case, the script will download the default model (a fine-tuned BART-base) automatically.
You can also use an alternative model by adding the parameter `--model_path`. The default model was
trained using the following command:
```
factorsum/scripts/run_summarization.py \
--model_name_or_path facebook/bart-base \
--do_train --do_eval --do_predict \
--output_dir output/legal-k_5_samples_20 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 4 \
--predict_with_generate \
--generation_max_length 128 \
--generation_num_beams 4 \
--val_max_target_length 128 \
--max_source_length 1024 \
--max_target_length 128 \
--fp16 \
--save_total_limit 2 \
--save_strategy steps \
--evaluation_strategy steps --save_steps 2000 \
--eval_steps 2000 --max_steps 50000 \
--learning_rate 5e-5 --report_to None \
--metric_for_best_model eval_rouge1_fmeasure \
--load_best_model_at_end \
--max_train_samples 4000000 \
--max_eval_samples 5000 \
--max_predict_samples 5000 \
--train_file /path/to/data/folder/legal-random_k_5_samples_20_train.csv \
--validation_file /path/to/data/folder/legal-random_k_5_samples_20_validation.csv 
--test_file /path/to/data/folder/legal_summarization/legal-random_k_5_samples_20_test.csv \
--text_column source --summary_column target --seed 17
```

## Longformer

Below is an exmple command line that can train longformer models on our dataset using `torchrun` on 4 NVIDIA A100 (80GB) GPUs with 128 effective batch size and maximum input length of 4096 tokens:

```
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
    longformer.py \
    --train_path /path/to/data/train.csv \
    --dev_path /path/to/data/valid.csv \
    --test_path /path/to/data/test.csv \
    --local_rank 0 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --epochs 50 \
    --max_input_length 4096 \
    --experiment_name 'legal-input-4096' \
    --filtered_source 'none' \
    --learning_rate 1e-4 \
    --fp16
```
If you would like to truncate the model input to use only the first 1024 tokens, you can modify `--max_input_length` to be 1024 and remember to also update the `--experiment_name`. If you would like to use the tail 1024 tokens, then you need to also add `--truncate_left` to the command:

```
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
    longformer.py \
    --train_path /path/to/data/train.csv \
    --dev_path /path/to/data/valid.csv \
    --test_path /path/to/data/test.csv \
    --local_rank 0 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --epochs 50 \
    --max_input_length 1024 \
    --truncate_left \
    --experiment_name 'legal-input-1024-tail' \
    --filtered_source 'none' \
    --learning_rate 1e-4 \
    --fp16
```

If you would like to train models with max input of 1024 tokens with oracle paragraph information (please refer to Section 4 in the paper for details), first you need to download these further processed data ([train](https://drive.google.com/file/d/12tdNRnnbovRA1E-YItkKxzEoXfunctBd/view?usp=sharing) | [val](https://drive.google.com/file/d/1eyt6ekCQuIQw7vs4cscwKgxLyqbmVPvN/view?usp=sharing) | [test](https://drive.google.com/file/d/1sa6xT14SiZRm7-X0cG0XK6GXhq_P8FrR/view?usp=sharing)) where the input has been processed (shortened to only contain oracle paragraphs). Alternatively, you can process the data yourself using the `process_data` function provided in the `util.py` file (an example is provided above). Then you can use the below command to train the model:

```
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
    longformer.py \
    --train_path /path/to/data/train.csv \
    --dev_path /path/to/data/valid.csv \
    --test_path /path/to/data/test.csv \
    --train_oracle_paragraph_path /path/to/data/paragraphs_train.csv \
    --dev_oracle_paragraph_path /path/to/data/paragraphs_valid.csv \
    --test_oracle_paragraph_path /path/to/data/paragraphs_test.csv \
    --local_rank 0 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --epochs 50 \
    --max_input_length 1024 \
    --experiment_name 'legal-input-1024-filtered-para' \
    --filtered_source 'paras' \
    --learning_rate 1e-4 \
    --fp16
```

## Llama 2
For Llama 2 model's experiment, we use this [LLM](https://github.com/thefonseca/llms) repo. You need to first follow the instruction to setup the repo, and then run the following command line:

```
python -m llms.summarizers.evaluation \
    --model_name llama-2-7b-chat \
    --model_checkpoint_path /path/to/your/llama2/model/ \
    --model_dtype float16 \
    --dataset_name /path/to/data/test.csv \
    --source_key text \
    --target_key summary \
    --model_user_prompt "Write a summary of the text above in 4 sentences" \
    --output_dir output
```

You can change `--model_name` to run experiments using other model configurations, e.g. `llama-2-13b-chat`, `llama-2-70b-chat`.
