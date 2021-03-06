#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random,multiprocessing

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version
from utils.process_func import * 

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

PATH_SCRATCH_CACHE = "/scratch/w/wluyliu/yananc/cache"

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )

    parser.add_argument(
        "--overwrite_cache", type=bool, default=True, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--debug_cnt",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='t5-base',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text1",
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default="text2",
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )

    parser.add_argument(
        "--tags_column",
        type=str,
        choices=['tags_coarse', 'tags_fine'],
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )


    parser.add_argument(
        "--local_files_only",
        action="store_true",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--binomial",
        type=float,
        default=1
    )


    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=7, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=99, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default='t5',
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args



def main():
    args = parse_args()
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # data_files = {}
    # if args.train_file is not None:
    #     data_files["train"] = args.train_file
    # if args.validation_file is not None:
    #     data_files["validation"] = args.validation_file
    # extension = args.train_file.split(".")[-1]
    # raw_datasets = load_dataset(extension, data_files=data_files, cache_dir='./cache')

    file_list = {}
    for dsn in ['dev','test','train']:
        file_list[dsn] = '/gpfs/fs0/scratch/w/wluyliu/yananc/few_nerd_supervised/{}.json'.format(dsn)
    raw_datasets = datasets.load_dataset('json', data_files=file_list, cache_dir='/scratch/w/wluyliu/yananc/cache')

    if args.debug_cnt > 0:     
        # processed_datasets['train_dev'] = datasets.concatenate_datasets([processed_datasets["train"], processed_datasets["dev"]])
        random.seed(args.seed)
        random_ids = random.sample(raw_datasets['train']['id'], args.debug_cnt)
        raw_datasets['train'] = raw_datasets['train'].filter(lambda example: example['id'] in random_ids, num_proc= multiprocessing.cpu_count())

    raw_datasets['test'] = datasets.concatenate_datasets([raw_datasets["test"], raw_datasets["dev"]])
    random.seed(args.seed)
    random_ids_test = random.sample(raw_datasets['test']['id'], 10240)
    raw_datasets['test'] = raw_datasets['test'].filter(lambda example: example['id'] in random_ids_test, num_proc= multiprocessing.cpu_count())

    raw_datasets.pop('dev')

    print("raw_datasets summary")
    print(raw_datasets)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=PATH_SCRATCH_CACHE, local_files_only=args.local_files_only)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=PATH_SCRATCH_CACHE, local_files_only=args.local_files_only)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, \
            cache_dir=PATH_SCRATCH_CACHE, local_files_only=args.local_files_only)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer,\
                         cache_dir=PATH_SCRATCH_CACHE, local_files_only=args.local_files_only)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config, cache_dir=PATH_SCRATCH_CACHE, local_files_only=args.local_files_only
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    print("columns==>", column_names)

    # Get the column names for input/target.

    # dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    # if args.text_column is None:
    #     text_column = column_names[0] #if dataset_columns is not None else column_names[0]
    # else:
    #     text_column = args.text_column
    #     if text_column not in column_names:
    #         raise ValueError(
    #             f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
    #         )

    # if args.summary_column is None:
    #     summary_column = column_names[1] #if dataset_columns is not None else column_names[1]
    # else:
    #     summary_column = args.summary_column
    #     if summary_column not in column_names:
    #         raise ValueError(
    #             f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
    #         )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False


    def t5_format(example):
        source_ll = []
        target_ll = []
        length = min(len(example['tokens']), len(tokenizer.additional_special_tokens) )
        mask_binomial = np.random.binomial(size=length, n=1, p = args.binomial)
        for i in range( length ):
            source_ll.append(tokenizer.additional_special_tokens[i] + example['tokens'][i] )
            if mask_binomial[i]:
                target_ll.append(tokenizer.additional_special_tokens[i] + example[args.tags_column][i] )
            else:
                target_ll.append(tokenizer.additional_special_tokens[i] + example['tokens'][i] )
        example['text1'] = ' '.join(source_ll)
        example['text2'] = ' '.join(target_ll)

        return example
        

    def preprocess_function(examples):
        inputs = examples[args.text_column]
        targets = examples[args.summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    dataset_ix = raw_datasets.map(map_func, 
                    batched=False,
                    num_proc= multiprocessing.cpu_count() ,
                    load_from_cache_file=not args.overwrite_cache, remove_columns=['tags'],
                    desc = "Running ix mapping ==>")

    processed_datasets_t5 = dataset_ix.map(t5_format, 
                    batched=False,
                    num_proc= multiprocessing.cpu_count() ,
                    load_from_cache_file=not args.overwrite_cache, 
                    desc = "Running t5 mapping ==>")


    processed_datasets = processed_datasets_t5.map(
        preprocess_function,
        batched=True,
        num_proc= multiprocessing.cpu_count() ,
        remove_columns=processed_datasets_t5["train"].column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer ===>",
    )


    train_dataset =  processed_datasets['train']
    test_dataset =  processed_datasets["test"]
    

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 8):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info("\n")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels





    def postprocess_text_ner(decoded_preds, decoded_labels):
        y_true = []
        y_pred = [] 

        for decoded_pred, decoded_label in zip(decoded_preds, decoded_labels):
            decoded_pred = decoded_pred.replace('</s>','').replace('<pad>', '')
            decoded_label = decoded_label.replace('</s>','').replace('<pad>', '')
            label_tokens, pred_tokens = [], []
            for i, j in zip(tokenizer.additional_special_tokens[:-1], tokenizer.additional_special_tokens[1:]):
                if i not in decoded_label or decoded_label.endswith(i):
                    continue

                ref_ner = decoded_label.split(i)[1].split(j)[0].strip()
                if not ref_ner:
                    # print(i, j, "===>", decoded_label)
                    # print("blank==>ref_ner")
                    ref_ner = 'O'
                if i in decoded_pred:
                    gen_ner = decoded_pred.split(i)[1].split(j)[0].strip()
                    if not gen_ner:
                        # print(i, j, "===>", decoded_pred)
                        gen_ner = 'O'
                        # print("blank==>gen_ner")
                else:
                    gen_ner = 'O'
                label_tokens.append(ref_ner)
                pred_tokens.append(gen_ner)

            assert len(label_tokens) == len(pred_tokens)
            y_true.append(label_tokens)
            y_pred.append(pred_tokens)
        return y_pred, y_true



    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    # metric = datasets.load_metric('rouge', cache_dir='/scratch/w/wluyliu/yananc/cache')
    metric_ner = datasets.load_metric('seqeval', cache_dir='/scratch/w/wluyliu/yananc/cache_gen_ner', experiment_id=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)))
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        if  epoch > 20:
            logger.info("begin to eval")
            model.eval()
            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": args.val_max_target_length if args is not None else config.max_length,
                "num_beams": args.num_beams,
            }

            decoded_preds_all = []
            decoded_labels_all = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

                    decoded_preds_all.extend(decoded_preds)
                    decoded_labels_all.extend(decoded_labels)

            assert len(decoded_preds_all) == len(decoded_labels_all)
            decoded_preds_, decoded_labels_ = postprocess_text_ner(decoded_preds_all, decoded_labels_all)
            print(random.sample(list(zip(decoded_labels_, decoded_preds_)), 32))
            metric_ner.add_batch(predictions=decoded_preds_, references=decoded_labels_)

            # result = metric.compute(use_stemmer=True)
            # # Extract a few results from ROUGE
            # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            # result = {k: round(v, 4) for k, v in result.items()}
            # logger.info(result)

            return_entity_level_metrics = False
            results = metric_ner.compute()
            # if return_entity_level_metrics:
            #     # Unpack nested dictionaries
            #     final_results = {}
            #     for key, value in results.items():
            #         if isinstance(value, dict):
            #             for n, v in value.items():
            #                 final_results[f"{key}_{n}"] = v
            #         else:
            #             final_results[key] = value
            #     return final_results
            # else:
            report = {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    # "accuracy": results["overall_accuracy"],
                }
            print("t5_ner_report ==>",  epoch, args.tags_column, args.max_target_length, args.debug_cnt, report)


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
        # epoch_output_dir = "{}/binomial_{}/epoch_{}".format(args.output_dir, args.binomial, epoch)
        # os.makedirs(epoch_output_dir, exist_ok=True)
        # unwrapped_model.save_pretrained(epoch_output_dir, save_function=accelerator.save)
    # return unwrapped_model, tokenizer


    processed_datasets_t5_gen = processed_datasets_t5.map(t5_format, 
                    batched=False,
                    num_proc= multiprocessing.cpu_count() ,
                    load_from_cache_file=not args.overwrite_cache, 
                    desc = "Running t5 mapping ==>")

    def clean_gen_span(span):
        for iden in tokenizer.additional_special_tokens + [tokenizer.unk_token, tokenizer.eos_token, tokenizer.pad_token]:
            span = span.replace(iden, '')
        return span.strip()


    processed_datasets_t5_shuffle = processed_datasets_t5_gen.shuffle()

    ii = 0 
    output_texts = []
    while ii <= len(processed_datasets_t5_shuffle['train']):
        text1s = processed_datasets_t5_shuffle['train'][ii:ii+args.per_device_eval_batch_size ]['text2']
        text2s = processed_datasets_t5_shuffle['train'][ii:ii+args.per_device_eval_batch_size ]['text1']
        if not text1s:
            break 
        text2s_ori = []
        for t in text2s:
            text2_decode = tokenizer.decode(tokenizer.encode(t), clean_up_tokenization_spaces=True, skip_special_tokens=True)
            text2s_ori.append(text2_decode)

        inputs = tokenizer(text1s, return_tensors='pt', padding=True, truncation=True)

        output = unwrapped_model.generate(input_ids=inputs['input_ids'].to(device), 
                       attention_mask=inputs['attention_mask'].to(device), do_sample=False, max_length=1024,
                       top_p=0.9, top_k=0, temperature=1.2 ) 

        output_decode = tokenizer.batch_decode(output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        output_decode_ori = tokenizer.batch_decode(output, clean_up_tokenization_spaces=True)

        output_texts.extend([dec.replace('</s>','').replace('<pad>','') for dec in output_decode_ori])
        
        # for l, p in zip(text2s_ori, output_decode): 
        #     print(l)
        #     print(p)
        #     print()

        print(ii, inputs['input_ids'].shape)

        ii += args.per_device_eval_batch_size 
        torch.cuda.empty_cache()

    assert len(output_texts) == len(processed_datasets_t5_shuffle['train'])


    with open('/scratch/w/wluyliu/yananc/few_nerd_supervised/da_coarse_binomal_{}_{}.json'.format(args.seed, args.binomial), 'w') as f:

        for ii, text2, text1, text_gen, tags in zip(  processed_datasets_t5_shuffle['train']['id'], \
                                                      processed_datasets_t5_shuffle['train']['text2'], \
                                                      processed_datasets_t5_shuffle['train']['text1'], \
                                                      output_texts, \
                                                      processed_datasets_t5_shuffle['train'][args.tags_column]):
            idens = []
            ix = 0
            for tag, i in zip(text2.split(), text1.split()):
                iden = "<extra_id_{}>".format(ix)
                iden_ = "<extra_id_{}>".format(ix+1)

                if iden in text_gen:
                    span = text_gen.split(iden)[1].split(iden_)[0]  
                    span = clean_gen_span(span)
                    if not span:
                        span = tokenizer.unk_token
                else:
                    span = tokenizer.unk_token

                print(tag.replace(iden, ''), '==>', i.replace(iden, ''), '--->', span)
                idens.append(span)
                ix += 1
            print(idens)
            dic = {}
            dic['id'] = ii
            dic['tokens'] = idens
            dic[args.tags_column] = tags[:len(idens)]
            assert len(dic[args.tags_column]) == len(dic['tokens'])
            print()

            json_string = json.dumps(dic)
            f.write(json_string+'\n')
            print('\n\n') 


if __name__ == "__main__":
    main()
