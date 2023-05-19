from transformers import AutoModelForQuestionAnswering, BertForSequenceClassification, AutoTokenizer, TrainingArguments, default_data_collator, DataCollatorWithPadding, Trainer
from datasets import DatasetDict, Value, load_metric, concatenate_datasets, load_from_disk, load_dataset, Dataset
from scipy.special import softmax
from tqdm.auto import tqdm
from torch import ge, nn
from torch.autograd import grad
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from typing import Dict, NamedTuple, Optional, Tuple, Union
from pathlib import Path
from difflib import SequenceMatcher
from copy import deepcopy
from numbers import Number
from collections import OrderedDict
import torch
import time
import math
import numpy as np
import collections
import pandas as pd
import operator


def load_tokenizer(task):
    if task != "squad":
        return AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        return AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')


def tokenize_data(task, dataset, tokenizer):
    if task == "sa":
        def tokenize_function(examples):
            return tokenizer(examples["test_case"], padding="max_length", max_length=128, truncation=True)
        return dataset.map(tokenize_function, batched=True)

    elif task == "qqp":
        return dataset.map(lambda e: tokenizer(e["question1"], e["question2"],
                                               truncation=True, padding=True))

    elif task == "squad":
        def preprocess_examples(examples, max_length=384, stride=128):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs["offset_mapping"]
            sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = answers[sample_idx]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                {inputs.setdefault(k, []).append(v[sample_idx])
                 for k, v in examples.items()}

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions

            return inputs
        try:
            cols_to_remove = dataset["train"].column_names
        except KeyError:
            cols_to_remove = dataset.column_names
        return dataset.map(preprocess_examples,
                           batched=True,
                           remove_columns=cols_to_remove)


def split_data_by_test_type(dataset):
    mft_dataset = dataset.filter(lambda x: x["type"] == "mft")
    dir_dataset = dataset.filter(lambda x: x["type"] == "dir")
    inv_dataset = dataset.filter(lambda x: x["type"] == "inv")
    return mft_dataset, dir_dataset, inv_dataset


def process_labels(task, data, iid=False):
    if task == "sa":
        if iid:
            labels_to_soft = [
                np.array([1., 0.]),
                np.array([0., 1.])
            ]
        else:
            labels_to_soft = {
                '0': np.array([1., 0.]),
                '1': np.array([0.5, 0.5]),
                '2': np.array([0., 1.]),
                'not_0': np.array([1/3, 2/3]),
            }

        def soft_to_label(soft):
            if np.allclose(soft, [1., 0.]):
                return "0"
            elif np.allclose(soft, [.5, .5]):
                return "1"
            if np.allclose(soft, [0., 1.]):
                return "2"
            if np.allclose(soft, [1/3, 2/3]):
                return "not_0"

        def process_labels(examples):
            labels = []
            for label in examples['label']:
                labels.append(labels_to_soft[label])
            return {'label': labels}

        return data.map(process_labels, batched=True), labels_to_soft, soft_to_label

    elif task == "qqp":
        return data.cast_column("label", Value('int64')), None, None

    elif task == "squad":
        return data, None, None


def align_inv(clean_datasets, id_to_samples):
    sm = SequenceMatcher()
    for clean_dataset in clean_datasets.values():
        for example in tqdm(clean_dataset):
            func = example["functionality"]
            test_id = example["test_id"]
            orig_ids = example["input_ids"]
            sm.set_seq2(orig_ids)
            for perturb in id_to_samples[func][test_id]:
                perturb_ids = perturb["input_ids"]
                sm.set_seq1(perturb_ids)
                matches = sm.get_matching_blocks()
                perturb_indxs = [i for x in matches for i in list(
                    range(x[0], x[0] + x[2]))]
                orig_indxs = [i for x in matches for i in list(
                    range(x[1], x[1] + x[2]))]
                assert len(perturb_indxs) == len(orig_indxs)
                size_mismatch = len(orig_ids) - len(orig_indxs)
                perturb_indxs = perturb_indxs + [0] * size_mismatch
                orig_indxs = orig_indxs + [0] * size_mismatch
                perturb["align_ids"] = (orig_indxs, perturb_indxs)


def extract_perturb(datasets):
    perturbs = DatasetDict()
    clean_datasets = DatasetDict()
    for split, dataset in datasets.items():
        base_cases_idxs = []
        perturb_cases_idxs = []
        current_func = None
        current_id = None
        i = 0
        for row in dataset:
            func, test_id = row["functionality"], row["test_id"]
            if func == current_func and test_id == current_id:
                perturb_cases_idxs.append(i)
            else:
                base_cases_idxs.append(i)
            current_func = func
            current_id = test_id
            i += 1
        clean_datasets[split] = datasets[split].select(base_cases_idxs)
        perturbs[split] = datasets[split].select(perturb_cases_idxs)
    return clean_datasets, perturbs


def get_id_to_samples(dataset, task):
    dic = {}
    if task == "qqp":
        dataset = dataset.remove_columns(["capability",
                                          "type", "direction", "slice", "question1",
                                          "question2"])
    for test_case in dataset:
        test_id, func = test_case["test_id"], test_case["functionality"]
        if task == "qqp":
            test_case.pop("functionality")
        dic.setdefault(func, {}).setdefault(test_id, []).append(test_case)
    return dic


def squash_to_neutral(preds):
    pr = preds[:, 1]
    pp = np.zeros((pr.shape[0], 3))
    margin_neutral = 1/3.
    mn = margin_neutral / 2.
    neg = pr < 0.5 - mn
    pp[neg, 0] = 1 - pr[neg]
    pp[neg, 2] = pr[neg]
    pos = pr > 0.5 + mn
    pp[pos, 0] = 1 - pr[pos]
    pp[pos, 2] = pr[pos]
    neutral_pos = (pr >= 0.5) * (pr < 0.5 + mn)
    pp[neutral_pos, 1] = 1 - (1 / margin_neutral) * \
        np.abs(pr[neutral_pos] - 0.5)
    pp[neutral_pos, 2] = 1 - pp[neutral_pos, 1]
    neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
    pp[neutral_neg, 1] = 1 - (1 / margin_neutral) * \
        np.abs(pr[neutral_neg] - 0.5)
    pp[neutral_neg, 0] = 1 - pp[neutral_neg, 1]
    preds = np.argmax(pp, axis=1)
    return preds


def get_model_init(task, model_name, dev=False, method=None):
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    if method == "dropout":
        hidden_dropout_prob *= 3
        attention_probs_dropout_prob *= 3

    if task == "sa":
        def model_init():
            model = BertForSequenceClassification.from_pretrained(str(model_name).replace(
                task, "SST2"), problem_type="multi_label_classification", hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob)
            if method in ["freeze", "lp-ft"]:
                for param in model.base_model.parameters():
                    param.requires_grad = False
            return model
    elif task == "qqp":
        def model_init():
            model = BertForSequenceClassification.from_pretrained(
                model_name, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob)
            if method in ["freeze", "lp-ft"]:
                for param in model.base_model.parameters():
                    param.requires_grad = False
            return model
    elif task == "squad":
        def model_init():
            if dev:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    "distilbert-base-uncased-distilled-squad")
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob)
            if method in ["freeze", "lp-ft"]:
                for param in model.base_model.parameters():
                    param.requires_grad = False
            return model
    return model_init


def get_compute_metric(task, test_type, soft_to_label=None):
    if test_type in ["mft", "iid"] and task == "sa":
        if test_type == "mft":
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = softmax(logits, axis=-1)
                preds = squash_to_neutral(preds)
                labels = [soft_to_label(l) for l in labels]
                n = 0
                c = 0
                for p, l in zip(preds, labels):
                    if l == "not_0":
                        if p == 1 or p == 2:
                            c += 1
                    else:
                        if p == int(l):
                            c += 1
                    n += 1
                return {"accuracy": c/n}
        else:
            metric = load_metric("glue", "sst2")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                labels = np.argmax(labels, axis=-1)
                return metric.compute(predictions=predictions, references=labels)
    elif test_type == "inv" and task != "squad":
        def compute_metrics(eval_pred, tol=0.1):
            perturb, orig = eval_pred
            perturb_probs = softmax(perturb, axis=-1)
            orig_probs = softmax(orig, axis=-1)
            return {"pass_inv": (np.abs(perturb_probs-orig_probs)[:, 1] <= tol).mean()}
    elif test_type == "dir" and task == "sa":
        def compute_metrics(eval_pred, tol=0.1):
            _, errors = eval_pred
            return {"pass_dir": (errors <= tol).mean()}
    elif task == "qqp" and test_type != "inv":
        metric = load_metric("glue", config_name="qqp")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            logits = logits.reshape((-1, 2))
            labels = labels.reshape((-1))
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
    elif task == "squad":
        if test_type in ["mft", "iid"]:
            def compute_metrics(eval_preds):
                logits, labels = eval_preds
                start_logits, end_logits = logits
                start_positions, end_positions = labels
                start_preds = np.argmax(start_logits, axis=-1)
                end_preds = np.argmax(end_logits, axis=-1)
                start_acc = (start_preds == start_positions).mean()
                end_acc = (end_preds == end_positions).mean()
                mean_acc = (start_acc + end_acc)/2
                return {"start_acc": start_acc, "end_acc": end_acc, "mean_acc": mean_acc}
        elif test_type == "inv":
            def compute_metrics(eval_preds, tol=0.1):
                (perturb_start_probs, orig_start_probs), (perturb_end_probs,
                                                          orig_end_probs) = eval_preds
                start_pass_inv = (
                    np.abs(perturb_start_probs-orig_start_probs) <= tol).mean()
                end_pass_inv = (
                    np.abs(perturb_end_probs-orig_end_probs) <= tol).mean()
                mean_pass_inv = (start_pass_inv + end_pass_inv)/2
                return {"start_pass_inv": start_pass_inv, "end_pass_inv": end_pass_inv, "mean_pass_inv": mean_pass_inv}
    return compute_metrics


def get_training_args(task, model_path, test_type, bs=4):
    if test_type == "mft":
        training_args = TrainingArguments(output_dir=model_path,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=True,
                                          learning_rate=2e-5,
                                          per_device_eval_batch_size=bs*2,
                                          per_device_train_batch_size=bs,
                                          save_strategy="no",
                                          seed=42
                                          )
    else:
        training_args = TrainingArguments(output_dir=model_path,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=True,
                                          learning_rate=2e-5,
                                          per_device_eval_batch_size=bs,
                                          per_device_train_batch_size=bs//2,
                                          save_strategy="no",
                                          seed=42,
                                          remove_unused_columns=False,
                                          )
    return training_args


def get_data_collator(task, test_type, id_to_samples=None, tokenizer=None):
    if task == "squad":
        class DataCollator:
            def __call__(self, examples: list[dict]):
                if test_type == "inv":
                    perturbed_samples = []
                    for e in examples:
                        func, test_id = e["functionality"], e["test_id"]
                        perturbed_sample = np.random.choice(
                            id_to_samples[func][test_id]).copy()
                        align_ids = perturbed_sample["align_ids"]
                        e["align_ids"] = align_ids[0]
                        perturbed_sample["align_ids"] = align_ids[1]
                        perturbed_samples.append(perturbed_sample)
                    batch = examples + perturbed_samples
                else:
                    batch = examples
                batch = [{k: v for k, v in s.items() if k in ["input_ids", "attention_mask", "start_positions", "end_positions", "align_ids"]}
                         for s in batch]
                return default_data_collator(batch)
        return DataCollator()

    elif task == "sa":
        if test_type == "mft":
            return default_data_collator
        else:
            class DataCollator:
                def __call__(self, examples: list[dict]):
                    perturb_samples = []
                    for e in examples:
                        func, test_id = e["functionality"], e["test_id"]
                        perturb_samples.append(np.random.choice(
                            id_to_samples[func][test_id]))
                    batch = examples + perturb_samples
                    return default_data_collator(batch)
            return DataCollator()
    elif task == "qqp":
        class DataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.collator = DataCollatorWithPadding(tokenizer=tokenizer)

            def __call__(self, examples: list[dict]):
                if test_type == "inv":
                    perturb_samples = []
                elif test_type == "dir":
                    batch = []
                for e in examples:
                    func, test_id = e["functionality"], e["test_id"]
                    e.pop("functionality")
                    if test_type == "inv":
                        perturb_sample = np.random.choice(
                            id_to_samples[func][test_id])
                        perturb_sample.copy().pop("label")
                        perturb_samples.append(perturb_sample)
                    elif test_type == "dir":
                        if func in ["Testing implications", "(q, paraphrase(q))"]:
                            batch.extend(
                                [e] + np.random.choice(id_to_samples[func][test_id], size=2, replace=False).tolist())
                        else:
                            options = id_to_samples[func][test_id]
                            if len(options) >= 3:
                                size = 3
                            else:
                                size = len(options)
                            batch.extend(np.random.choice(
                                options, size=size, replace=False).tolist())
                if test_type == "inv":
                    batch = examples + perturb_samples
                elif test_type == "mft":
                    batch = examples
                return dict(self.collator(batch))
        return DataCollator(tokenizer)


def evaluate(trainer, task, compute_metrics, datasets, split, clean_datasets=None, perturb_datasets=None, test_type="mft", soft_to_label=None):
    if task == "squad":
        if test_type == "inv":
            model = trainer.model
            trainer = Trainer(model=model)
        predictions, _, _ = trainer.predict(datasets[split])
        start_logits, end_logits = predictions
        if test_type == "mft":
            print(compute_metrics(start_logits, end_logits,
                  datasets[split], clean_datasets[split]))
        else:
            ids = concatenate_datasets(clean_datasets.values())["id"]
            results, preds, labels = compute_metrics(
                start_logits, end_logits, datasets[split], clean_datasets[split], original_ids=ids, return_answers=True)
            print(results)
            hits = [p["prediction_text"] == l["answers"]["text"][0]
                    for p, l in zip(preds, labels)]
            for i, hit in enumerate(hits):
                if hit == False:
                    sample = perturb_datasets[split][i]
                    print(sample["context"], sample["question"],
                          preds[i], labels[i])
                    print(sample["functionality"])

    else:
        if task == "sa":
            if test_type == "mft":
                print(trainer.evaluate(datasets[split]))
                preds, labels, _ = trainer.predict(datasets[split])
                hits = preds.argmax(axis=-1) == labels
                preds_s = softmax(preds, axis=-1)
                preds_s = squash_to_neutral(preds_s)
                labels_h = [soft_to_label(l) for l in labels]
                i = 0
                for p, l in zip(preds_s, labels_h):
                    if l == "not_0":
                        if p == 0:
                            print(datasets[split][i]["test_case"], p, l)
                    else:
                        if p != int(l):
                            print(datasets[split][i]["test_case"], p, l)
                    i += 1
            elif test_type == "inv":
                print(trainer.evaluate(clean_datasets[split]))
                preds, labels, _ = trainer.predict(clean_datasets[split])
                hits = preds.argmax(axis=-1) == labels.argmax(axis=-1)
                for i in range(len(clean_datasets[split])):
                    if hits[i] == False:
                        print(clean_datasets[split][i]["functionality"], softmax(
                            preds[i]), softmax(labels[i]))
            elif test_type == "dir":
                print(trainer.evaluate(clean_datasets[split]))
                errors, _, _ = trainer.predict(clean_datasets[split])
                real_errors = errors > 0.1
                for i in range(len(clean_datasets[split])):
                    if real_errors[i]:
                        print(clean_datasets[split][i]
                              ["functionality"], errors[i])

        elif task == "qqp":
            columns_to_remove = ["label", "capability",
                                 "type", "direction", "slice", "question1",
                                 "question2"]
            if test_type == "mft":
                print(trainer.evaluate(datasets[split]))
                preds, labels, _ = trainer.predict(datasets[split])
                hits = preds.argmax(axis=-1) == labels
                for i, h in enumerate(hits.tolist()):
                    if not h:
                        print(datasets[split][i]["question1"], datasets[split][i]["question2"],
                              softmax(preds[i], axis=-1), labels[i])
            elif test_type == "inv":
                print(trainer.evaluate(
                    clean_datasets[split].remove_columns(columns_to_remove)))
                preds, labels, _ = trainer.predict(
                    clean_datasets[split].remove_columns(columns_to_remove))
                hits = preds.argmax(axis=-1) == labels.argmax(-1)
                for i in range(len(clean_datasets[split])):
                    if hits[i] == False:
                        print(softmax(preds[i], axis=-1), softmax(labels[i],
                              axis=-1), clean_datasets[split][i]["functionality"])
            elif test_type == "dir":
                print(trainer.evaluate(
                    clean_datasets[split].remove_columns(columns_to_remove[1:])))
                preds, labels, _ = trainer.predict(
                    clean_datasets[split].remove_columns(columns_to_remove[1:]))
                preds = preds.reshape((-1, 2))
                labels = labels.reshape((-1))
                hits = preds.argmax(axis=-1) == labels
                hits
                for i, h in enumerate(hits.tolist()):
                    if not h:
                        print(softmax(preds[i], axis=-1), labels[i],
                              clean_datasets[split][i//3]["functionality"])


def process_dir_datasets(task, datasets):
    if task == "qqp":
        return datasets.cast_column("label", Value("uint8")), None, None
    elif task == "sa":
        direction_to_int = {k: v for v, k in enumerate(
            np.unique(datasets["train"]["direction"]))}
        int_to_direction = [k for k, _ in direction_to_int.items()]

        def process_directions(examples):
            directions = []
            for direction in examples['direction']:
                directions.append(direction_to_int[direction])
            return {'direction': directions}
        return datasets.map(process_directions, batched=True), direction_to_int, int_to_direction
    elif task == "squad":
        return None, None, None


def get_squad_preds(test_type):
    metric = load_metric("squad")

    def compute_metrics(start_logits, end_logits, features, examples, original_ids=None, n_best=20, max_answer_length=30, return_answers=False):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["id"]].append(idx)

        predicted_answers = []
        if test_type != "mft":
            orig_answer_dict = {}

        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(
                    start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(
                    end_logit)[-1: -n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if (not offsets[start_index]) or (not offsets[end_index]):
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if test_type == "inv":
                func = example["functionality"]
                test_id = example["test_id"]
                if len(answers) > 0:
                    best_answer = max(
                        answers, key=lambda x: x["logit_score"])
                    if example_id not in original_ids:
                        predicted_answers.append(
                            {"id": example_id,
                                "prediction_text": best_answer["text"]}
                        )
                    else:
                        orig_answer_dict.setdefault(
                            func, {})[test_id] = best_answer["text"]
                else:
                    if example_id not in original_ids:
                        predicted_answers.append(
                            {"id": example_id, "prediction_text": ""})
                    else:
                        orig_answer_dict.setdefault(func, {})[test_id] = ""

            else:
                if len(answers) > 0:
                    best_answer = max(
                        answers, key=lambda x: x["logit_score"])
                    predicted_answers.append(
                        {"id": example_id,
                            "prediction_text": best_answer["text"]}
                    )
                else:
                    predicted_answers.append(
                        {"id": example_id, "prediction_text": ""})

        if test_type == "inv":
            theoretical_answers = [{"id": ex["id"], "answers": {"text": [orig_answer_dict[ex["functionality"]]
                                                                         [ex["test_id"]]], "answer_start": [None]}} for ex in examples if ex["id"] not in original_ids]
        else:
            theoretical_answers = [
                {"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers) if not return_answers else (metric.compute(predictions=predicted_answers, references=theoretical_answers), predicted_answers, theoretical_answers)
    return compute_metrics


class DataLoaderWithTestType:
    """
    Wrapper around a DataLoader to also yield the test_type
    """

    def __init__(self, test_type, data_loader, env=None):
        self.test_type = test_type
        self.data_loader = data_loader
        self.env = env

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["test_type"] = StrIgnoreDevice(self.test_type)
            if self.env:
                batch["env"] = StrIgnoreDevice(self.env)
            yield batch


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `test_type` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class MultiTestTypeDataloader:
    """
    Data loader that combines and samples from multiple single-test_type
    data loaders.
    """

    def __init__(self, dataloader_dict, method=None, eval_mode=False, bs=None, meta_steps=5):
        self.dataloader_dict = dataloader_dict
        self.eval_mode = eval_mode
        self.method = method
        self.meta_steps = meta_steps

        self.num_batches_dict = {
            test_type: len(dataloader)
            for test_type, dataloader in self.dataloader_dict.items()
        }
        self.test_type_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        if self.method == "irm":
            n_samples_per_batch = sum(
                [v.batch_size for v in self.dataloader_dict.values()])
            n_samples = sum(
                [len(v)*v.batch_size for v in self.dataloader_dict.values()])

            return math.ceil(n_samples / n_samples_per_batch)
        elif self.method in ["dro", "fish"]:
            avg_batch_size = np.mean(
                [v.batch_size for v in self.dataloader_dict.values()])
            if self.method == "fish":
                avg_batch_size *= self.meta_steps
            return math.ceil(len(self.dataset)/avg_batch_size)
        else:
            return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a test_type, and yield a batch from the respective
        test_type Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        if self.eval_mode or self.method not in ["irm", "dro", "fish"]:
            test_type_choice_list = []
            for i, test_type in enumerate(self.test_type_list):
                test_type_choice_list += [i] * self.num_batches_dict[test_type]
            test_type_choice_list = np.array(test_type_choice_list)
            if not self.eval_mode:
                np.random.shuffle(test_type_choice_list)
            dataloader_iter_dict = {
                test_type: iter(dataloader)
                for test_type, dataloader in self.dataloader_dict.items()
            }
            for test_type_choice in test_type_choice_list:
                test_type = self.test_type_list[test_type_choice]
                yield next(dataloader_iter_dict[test_type])
        else:
            test_type_choice_list = self.test_type_list
            for i in range(len(self)):
                np.random.shuffle(test_type_choice_list)
                dataloader_iter_dict = {
                    test_type: iter(dataloader)
                    for test_type, dataloader in self.dataloader_dict.items()
                }
                if self.method in ["irm", "fish"]:
                    env_batches = []
                    if self.method == "irm":
                        envs = test_type_choice_list
                    elif self.method == "fish":
                        envs = test_type_choice_list[:self.meta_steps]
                    for test_type_choice in envs:
                        env_batches.append(
                            next(dataloader_iter_dict[test_type_choice]))
                    yield env_batches
                elif self.method == "dro":
                    yield next(dataloader_iter_dict[test_type_choice_list[0]])


def get_sa_dir_errors(self, logits, half, directions):
    orig_probs = self.softmax(logits[:half])
    perturb_probs = self.softmax(logits[half:])
    directions = directions[:half].unsqueeze(0)
    preds = orig_probs.argmax(axis=-1, keepdims=True)

    diffs = perturb_probs - orig_probs
    not_less_neg = self.relu(-diffs[:, 0])
    not_less_pos = self.relu(-diffs[:, 1])
    not_less_conf = self.relu(-torch.gather(diffs, -1, preds)).squeeze()
    not_more_conf = self.relu(torch.gather(diffs, -1, preds)).squeeze()
    all_diffs = torch.vstack(
        [not_less_neg, not_less_pos, not_less_conf, not_more_conf])
    errors = torch.take_along_dim(
        all_diffs, directions, 0).view(-1)
    return errors


def get_squad_inv_aligns(start_logits, end_logits, align_ids, half):
    start_logits_aligned = torch.gather(start_logits, -1, align_ids)
    end_logits_aligned = torch.gather(end_logits, -1, align_ids)
    orig_start_logits_aligned = start_logits_aligned[:half]
    orig_end_logits_aligned = end_logits_aligned[:half]
    perturb_start_logits_aligned = start_logits_aligned[half:]
    perturb_end_logits_aligned = end_logits_aligned[half:]
    return orig_start_logits_aligned, orig_end_logits_aligned, perturb_start_logits_aligned, perturb_end_logits_aligned


def get_compute_loss(self, test_type, get_dummy_loss=False):
    task = self.task
    if task == "sa":
        if test_type in ["mft", "iid"]:
            def compute_loss_fct(self, model, inputs, return_outputs=False):
                inputs = {k: v for k, v in inputs.items() if k in [
                    "input_ids", "token_type_ids", "attention_mask", "labels"]}
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                loss = outputs.get("loss")
                if get_dummy_loss:
                    dummy_loss = self.cross_loss(outputs.get(
                        "logits") * self.dummy_classifier, labels)
                    loss = (loss, dummy_loss)
                return (loss, outputs) if return_outputs else loss
        elif test_type == "dir":
            def compute_loss_fct(self, model, inputs, return_outputs=False):
                directions = inputs["direction"]
                inputs = {k: v for k, v in inputs.items() if k in [
                    "input_ids", "token_type_ids", "attention_mask"]}
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = self.bce_loss
                half = logits.shape[0]//2
                errors = get_sa_dir_errors(self, logits, half, directions)

                loss = loss_fct(errors, torch.zeros(
                    errors.shape).to(self.device))
                if get_dummy_loss:
                    dummy_loss = self.bce_loss(get_sa_dir_errors(
                        self, logits * self.dummy_classifier, half, directions), torch.zeros(errors.shape).to(self.device))
                    loss = (loss, dummy_loss)
                return (loss, errors) if return_outputs else loss

    elif task == "qqp":
        if test_type != "inv":
            def compute_loss_fct(self, model, inputs, return_outputs=False):
                inputs = {k: v for k, v in inputs.items() if k in [
                    "input_ids", "token_type_ids", "attention_mask", "labels"]}
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                loss = outputs.get("loss")
                if get_dummy_loss:
                    dummy_loss = self.cross_loss(outputs.get(
                        "logits") * self.dummy_classifier, labels)
                    loss = (loss, dummy_loss)
                return (loss, outputs) if return_outputs else loss
    elif task == "squad":
        if test_type in ["mft", "iid"]:
            def compute_loss_fct(self, model, inputs, return_outputs=False):
                inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "start_positions",
                                                                   "end_positions"]}
                # forward pass
                outputs = model(**inputs)
                start_logits, end_logits = outputs.get(
                    "start_logits"), outputs.get("end_logits")
                total_loss = outputs.get("loss")
                if get_dummy_loss:
                    dummy_start_loss = self.cross_loss(
                        start_logits * self.dummy_classifier, inputs.get("start_positions"))
                    dummy_end_loss = self.cross_loss(
                        end_logits*self.dummy_classifier, inputs.get("end_positions"))
                    total_loss = (
                        total_loss, (dummy_start_loss + dummy_end_loss)/2)
                return (total_loss, outputs) if return_outputs else total_loss

        elif test_type == "inv":
            def compute_loss_fct(self, model, inputs, return_outputs=False):
                align_ids = inputs["align_ids"]
                inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "start_positions",
                                                                   "end_positions"]}
                # forward pass
                outputs = model(**inputs)
                start_logits, end_logits = outputs.get(
                    "start_logits"), outputs.get("end_logits")
                half = start_logits.shape[0]//2
                loss_fct = self.cross_loss
                orig_start_logits_aligned, orig_end_logits_aligned, perturb_start_logits_aligned, perturb_end_logits_aligned = get_squad_inv_aligns(
                    start_logits, end_logits, align_ids, half)
                start_loss = loss_fct(
                    perturb_start_logits_aligned, self.softmax(orig_start_logits_aligned))
                end_loss = loss_fct(perturb_end_logits_aligned,
                                    self.softmax(orig_end_logits_aligned))
                total_loss = (start_loss + end_loss) / 2
                if get_dummy_loss:
                    dummy_start_loss = loss_fct(
                        perturb_start_logits_aligned * self.dummy_classifier, self.softmax(orig_start_logits_aligned))
                    dummy_end_loss = loss_fct(
                        perturb_end_logits_aligned * self.dummy_classifier, self.softmax(orig_end_logits_aligned))
                    total_loss = (
                        total_loss, (dummy_start_loss + dummy_end_loss)/2)
                perturb_start_preds = perturb_start_logits_aligned.argmax(-1)
                perturb_end_preds = perturb_end_logits_aligned.argmax(-1)
                orig_start_preds = orig_start_logits_aligned.argmax(-1)
                orig_end_preds = orig_end_logits_aligned.argmax(-1)
                return (total_loss, ((perturb_start_preds, orig_start_preds),
                                     (perturb_end_preds, orig_end_preds))) if return_outputs else total_loss

    if test_type == "inv":
        if task != "squad":
            def compute_loss_fct(self, model, inputs, return_outputs=False):
                inputs = {k: v for k, v in inputs.items() if k in [
                    "input_ids", "token_type_ids", "attention_mask"]}
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = self.cross_loss
                half = logits.shape[0]//2
                loss = loss_fct(logits[half:], self.softmax(logits[:half]))
                if get_dummy_loss:
                    dummy_loss = loss_fct(
                        logits[half:] * self.dummy_classifier, self.softmax(logits[:half]))
                    loss = (loss, dummy_loss)
                return (loss, outputs) if return_outputs else loss
    return compute_loss_fct


def get_prediction_step(self, test_type):
    task = self.task
    if task == "sa":
        if test_type in ["mft", "iid"]:
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                return (loss, outputs.get("logits"), inputs.get("labels"))

        elif test_type == "dir":
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, errors = self.compute_loss(
                        model, inputs, return_outputs=True)
                return (loss, errors, errors)

        elif test_type == "inv":
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                logits = outputs.get("logits")
                half = logits.shape[0]//2
                return (loss, logits[half:], logits[:half])
    elif task == "qqp":
        if test_type != "inv":
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                return (loss, outputs.get("logits"), inputs.get("labels"))
        else:
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                logits = outputs.get("logits")
                half = logits.shape[0]//2
                return (loss, logits[half:], logits[:half])
    elif task == "squad":
        if test_type in ["mft", "iid"]:
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                start_logits, end_logits = outputs.get(
                    "start_logits"), outputs.get("end_logits")
                start_positions, end_positions = inputs.get(
                    "start_positions"), inputs.get("end_positions")
                return (loss, (start_logits, end_logits), (start_positions, end_positions))
        else:
            def prediction_step_fct(self, model, inputs,  prediction_loss_only=False, ignore_keys=False):
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True)
                (perturbed_start_preds, orig_start_preds), (perturbed_end_preds,
                                                            orig_end_preds) = outputs
                return (loss, (perturbed_start_preds, orig_start_preds), (perturbed_end_preds, orig_end_preds))
    return prediction_step_fct


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class MultiTestTypeTrainer(Trainer):
    def __init__(self, task, method, *args, **kwargs):
        self.task = task
        self.method = method
        super().__init__(*args, **kwargs)
        self.compute_metrics_dic = self.compute_metrics
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.cross_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.device = self.model.device
        if self.method in ["irm", "dro", "fish"]:
            envs_by_type = {}
            datasets = {}
            for test_type, dataset in self.train_dataset.items():
                func_idxs = {}
                i = 0
                for row in tqdm(dataset):
                    if test_type != "iid":
                        key = row["functionality"]
                    else:
                        key = "iid"
                    func_idxs.setdefault((test_type, key), []).append(i)
                    i += 1
                envs_by_type[test_type] = {
                    k: dataset.select(v) for k, v in func_idxs.items()}
            for key in envs_by_type.keys():
                datasets = {**datasets, **envs_by_type[key]}
            self.train_dataset = datasets
            if self.method == "irm":
                if self.task != "squad":
                    num_labels = self.model.num_labels
                else:
                    num_labels = len(list(self.train_dataset.values())[
                                     0]["input_ids"][1])

                self.dummy_classifier = torch.nn.Parameter(
                    torch.ones(1, num_labels)).to(self.device)
                self.reg = 1e4
            elif self.method == "dro":
                self.n_groups = len(datasets)
                # self.group_counts = {k[1]: v.num_rows for k, v in self.train_dataset.items()}
                self.group2idx = {k[1]: i for k, i in zip(
                    self.train_dataset.keys(), range(self.n_groups))}
                self.step_size = 1e-2
                self.group_weights = torch.ones(
                    self.n_groups).cuda()/self.n_groups
            elif self.method == "fish":
                self.meta_steps = 5

    def get_single_dataloader(self, key, dataset, eval_mode=False):
        """
        Create a single-test_type data loader
        """
        if self.method in ["irm", "dro", "fish"] and not eval_mode:
            test_type = key[0]
            env = key[1]
        else:
            test_type = key
            env = None
        if not eval_mode:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            batch_size = self.args.train_batch_size
            sampler = (
                RandomSampler(dataset)
                if self.args.local_rank == -1
                else DistributedSampler(dataset)
            )
        else:
            batch_size = self.args.eval_batch_size
            sampler = (
                SequentialSampler(dataset)
                if self.args.local_rank == -1
                else DistributedSampler(dataset, shuffle=False)
            )

        if test_type in ["inv", "dir"]:
            if self.task == "qqp" and test_type == "dir":
                batch_size = batch_size // 3
            else:
                batch_size = batch_size // 2

        data_collator = self.data_collator[test_type]

        data_loader = DataLoaderWithTestType(
            test_type=test_type,
            env=env,
            data_loader=DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultiTestTypeDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        test_type Dataloader
        """
        meta_steps = None
        if self.method == "fish":
            meta_steps = self.meta_steps
        return MultiTestTypeDataloader({
            test_type: self.get_single_dataloader(test_type, test_type_dataset)
            for test_type, test_type_dataset in self.train_dataset.items()
        }, method=self.method, bs=self.args.train_batch_size, meta_steps=meta_steps)

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return {
            test_type: self.get_single_dataloader(
                test_type, test_type_dataset, eval_mode=True)
            for test_type, test_type_dataset in eval_dataset.items()
        }

    def get_test_dataloader(self, test_dataset):
        return {
            test_type: self.get_single_dataloader(
                test_type, test_type_dataset, eval_mode=True)
            for test_type, test_type_dataset in test_dataset.items()
        }

    def floating_point_ops(self, inputs):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.
        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs) if self.method not in ["irm", "fish"] else np.sum([self.model.floating_point_ops(batch) for batch in inputs])
        else:
            return 0

    def compute_loss(self, model, inputs, return_outputs=False, get_dummy_loss=False):
        test_type = inputs["test_type"]
        compute_loss_fct = get_compute_loss(
            self, test_type, get_dummy_loss=get_dummy_loss)
        return compute_loss_fct(self, model, inputs, return_outputs=return_outputs)

    def training_step(self, model, inputs):
        model.train()
        if self.method not in ["irm", "fish"]:
            inputs = self._prepare_inputs(inputs)
            loss = self.compute_loss(model, inputs)
            if self.method == "dro":
                group = inputs.get("env")
                group_idx = self.group2idx[group]
                self.group_weights[group_idx] = self.group_weights[group_idx] * \
                    torch.exp(self.step_size*loss.data)
                self.group_weights = self.group_weights / \
                    (self.group_weights.sum())
                loss = loss * self.group_weights[group_idx]
            loss.backward()
        else:
            loss = torch.tensor(0.0)
            if self.method == "irm":
                thresh = self.args.num_train_epochs/5
                if self.state.epoch < thresh:
                    reg = 1.0
                else:
                    reg = self.reg
                for batch_e in inputs:
                    batch_e = self._prepare_inputs(batch_e)
                    erm_loss, dummy_loss = self.compute_loss(
                        model, batch_e, get_dummy_loss=True)
                    penalty = (
                        (grad(dummy_loss, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()

                    batch_loss = erm_loss + reg * penalty
                    if reg > 1.0:
                        batch_loss /= reg
                    batch_loss.backward()
                    loss += batch_loss.detach().item()
            elif self.method == "fish":
                model_outer = deepcopy(model)
                if self.opt_state is not None:
                    self.optimizer.load_state_dict(self.opt_state)
                    self.lr_scheduler.load_state_dict(self.lr_state)
                    del self.opt_state
                    del self.lr_state
                    torch.cuda.empty_cache()
                for batch_e in inputs:
                    batch_e = self._prepare_inputs(batch_e)
                    batch_loss = self.compute_loss(model, batch_e)
                    batch_loss.backward()
                    loss += batch_loss.detach().item()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()
                meta_weights = fish_step(meta_weights=model_outer.state_dict(),
                                         inner_weights=model.state_dict(),
                                         meta_lr=5e-7 / self.args.learning_rate / self.meta_steps)
                model.load_state_dict(deepcopy(meta_weights))
                del meta_weights
                del model_outer
                torch.cuda.empty_cache()
                self.opt_state = deepcopy(self.optimizer.state_dict())
                self.lr_state = deepcopy(self.lr_scheduler.state_dict())
            loss /= len(inputs)
        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=False):
        test_type = inputs["test_type"]
        prediction_step_fct = get_prediction_step(self, test_type)
        return prediction_step_fct(self, model, inputs,  prediction_loss_only=prediction_loss_only,
                                   ignore_keys=ignore_keys)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        outputs = {}

        for test_type, dl in eval_dataloader.items():
            start_time = time.time()
            self.compute_metrics = self.compute_metrics_dic[test_type]

            output = eval_loop(
                dl,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{test_type}",
            )
            outputs[test_type] = output

            total_batch_size = dl.batch_size * self.args.world_size

            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)

            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)

        return {test_type: output.metrics for test_type, output in outputs.items()}

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        outputs = {}

        for test_type, dl in test_dataloader.items():
            start_time = time.time()
            self.compute_metrics = self.compute_metrics_dic[test_type]
            output = eval_loop(
                dl,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{test_type}",
            )
            outputs[test_type] = output

            total_batch_size = dl.batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self._memory_tracker.stop_and_update_metrics(output.metrics)

        return {test_type: PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics) for test_type, output in outputs.items()}

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        # if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
        #     # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
        #     optimizer = self.optimizer.optimizer
        # else:
        #     optimizer = self.optimizer
        if self.method == "fish":
            num_training_steps *= self.meta_steps
            self.opt_state = None
            self.lr_state = None
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=self.optimizer)


def get_compute_objective(task):
    if task == "sa":
        def compute_objective(metrics):
            return metrics["mft"]["eval_mft_accuracy"] + metrics["inv"]["eval_inv_pass_inv"] + metrics["dir"]["eval_dir_pass_dir"]
    elif task == "qqp":
        def compute_objective(metrics):
            return (metrics["mft"]["eval_mft_accuracy"] + metrics["mft"]["eval_mft_f1"])/2 + metrics["inv"]["eval_inv_pass_inv"] + (metrics["dir"]["eval_dir_accuracy"] + metrics["dir"]["eval_dir_f1"])/2
    else:
        def compute_objective(metrics):
            return metrics["mft"]["eval_mft_mean_acc"] + metrics["inv"]["eval_inv_mean_pass_inv"]
    return compute_objective


def process_data(task, data_path, plus_iid, dev):
    dataset = load_from_disk(data_path)

    tokenizer = load_tokenizer(task)

    if dev:
        dataset = dataset.filter(lambda x: x["test_id"] in range(50))

    tokenized_datasets = tokenize_data(task, dataset, tokenizer)

    mft_datasets, dir_datasets, inv_datasets = split_data_by_test_type(
        tokenized_datasets)

    mft_datasets, labels_to_soft, soft_to_label = process_labels(
        task, mft_datasets)

    clean_inv_datasets, perturb_inv_datasets = extract_perturb(inv_datasets)

    perturb_inv_dataset = concatenate_datasets(perturb_inv_datasets.values())

    id_to_samples_inv = get_id_to_samples(perturb_inv_dataset, task)

    dir_datasets, direction_to_int, int_to_direction = process_dir_datasets(
        task, dir_datasets)

    if task != "squad":
        clean_dir_datasets, perturb_dir_datasets = extract_perturb(
            dir_datasets)
        perturb_dir_dataset = concatenate_datasets(
            perturb_dir_datasets.values())
        id_to_samples_dir = get_id_to_samples(perturb_dir_dataset, task)
    else:
        id_to_samples_dir = None

    train_dataset = {
        "mft": mft_datasets["train"],
        "inv": clean_inv_datasets["train"],
    }
    if task != "squad":
        train_dataset["dir"] = clean_dir_datasets["train"]

    eval_dataset = {
        "mft": mft_datasets["validation"],
        "inv": clean_inv_datasets["validation"]
    }
    if task != "squad":
        eval_dataset["dir"] = clean_dir_datasets["validation"]
    if task == "squad":
        align_inv(clean_inv_datasets, id_to_samples_inv)

    if plus_iid:
        if task == "sa":
            dataset = load_dataset("glue", "sst2")
            dataset = dataset.rename_column("sentence", "test_case")
        elif task == "qqp":
            dataset = load_dataset("glue", "qqp")
        elif task == "squad":
            dataset = load_dataset("squad")
        tokenized_dataset = tokenize_data(task, dataset, tokenizer)
        if "test" in tokenized_dataset.keys():
            del tokenized_dataset["test"]
        tokenized_dataset, _, _ = process_labels(
            task, tokenized_dataset, iid=True)
        val_test_datasets = tokenized_dataset["validation"].train_test_split(
            test_size=0.5, seed=42)
        val_iid = val_test_datasets["train"]
        train_iid = tokenized_dataset["train"]
        if dev:
            val_iid = val_iid.select(list(range(100)))
            train_iid = train_iid.select(list(range(100)))
        train_dataset["iid"] = train_iid
        eval_dataset["iid"] = val_iid

    return train_dataset, eval_dataset, tokenizer, id_to_samples_inv, id_to_samples_dir, soft_to_label, inv_datasets, clean_inv_datasets, perturb_inv_datasets


def get_trainer(task, tokenizer, training_args, train_dataset, eval_dataset, id_to_samples_inv, id_to_samples_dir, soft_to_label, pretrained_path,  plus_iid=False, dev=True, method=None):
    data_collators = {
        "mft": get_data_collator(task, "mft", tokenizer=tokenizer),
        "inv": get_data_collator(task, "inv", id_to_samples_inv, tokenizer),
    }
    if task != "squad":
        data_collators["dir"] = get_data_collator(
            task, "dir", id_to_samples_dir, tokenizer)

    compute_metrics = {
        "mft": get_compute_metric(task, "mft", soft_to_label),
        "inv": get_compute_metric(task, "inv", soft_to_label),
    }
    if task != "squad":
        compute_metrics["dir"] = get_compute_metric(task, "dir", soft_to_label)

    if plus_iid:
        data_collator = default_data_collator if task != "qqp" else DataCollatorWithPadding(
            tokenizer=tokenizer)
        data_collators["iid"] = lambda data: dict(data_collator(data))

        compute_metrics["iid"] = get_compute_metric(task, "iid", soft_to_label)

    model_init = get_model_init(task, pretrained_path, dev, method)

    if task == "qqp":
        columns_to_remove = ["label", "capability",
                             "type", "direction", "slice", "question1",
                             "question2"]
        train_dataset = train_dataset.copy()
        eval_dataset = eval_dataset.copy()
        for k, v in train_dataset.items():
            if k == "iid":
                cols = columns_to_remove[-2:]
            elif k == "inv":
                cols = columns_to_remove
            else:
                cols = columns_to_remove[1:]
            train_dataset[k] = v.remove_columns(cols)
        for k, v in eval_dataset.items():
            if k == "iid":
                cols = columns_to_remove[-2:]
            elif k == "inv":
                cols = columns_to_remove
            else:
                cols = columns_to_remove[1:]
            eval_dataset[k] = v.remove_columns(cols)

    elif task == "squad":
        columns_to_remove = ['id', 'title', 'context', 'question', 'answers']
        train_dataset = train_dataset.copy()
        eval_dataset = eval_dataset.copy()
        if plus_iid:
            train_dataset["iid"] = train_dataset["iid"].remove_columns(
                columns_to_remove)
            eval_dataset["iid"] = eval_dataset["iid"].remove_columns(
                columns_to_remove)

    trainer = MultiTestTypeTrainer(
        task=task,
        model_init=model_init,
        args=training_args,
        data_collator=data_collators,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        method=method
    )
    if method == "l2":
        trainer.args.weight_decay = 0.1
    return trainer


def get_iid_test(task, tokenizer, dev=False, return_val=False):
    if task == "sa":
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.rename_column("sentence", "test_case")
    elif task == "qqp":
        dataset = load_dataset("glue", "qqp")
    elif task == "squad":
        dataset = load_dataset("squad")

    val_test_datasets = dataset["validation"].train_test_split(
        test_size=0.5, seed=42)
    val_dataset = val_test_datasets["train"]
    test_dataset = val_test_datasets["test"]
    if dev:
        test_dataset = test_dataset.select(list(range(100)))
    if task != "squad":
        tokenized_test = tokenize_data(task, test_dataset, tokenizer)
        tokenized_val = tokenize_data(task, val_dataset, tokenizer)
    else:
        tokenized_test = load_from_disk("./data/squad/squad_target_data/test")
        tokenized_val = load_from_disk("./data/squad/squad_target_data/val")
        tokenized_test = tokenized_test.rename_column("example_id", "id")
        tokenized_val = tokenized_val.rename_column("example_id", "id")
        if dev:
            tokenized_test = tokenized_test.filter(
                lambda x: x["id"] in test_dataset["id"])
    if return_val:
        return (tokenized_val, tokenized_test) if task != "squad" else ((tokenized_val, val_dataset), (tokenized_test, test_dataset))
    return tokenized_test if task != "squad" else (tokenized_test, test_dataset)


def get_suite_test(task, tokenizer, dev=False, return_train_val=False):
    test_dataset = load_from_disk(f"./data/{task}/{task}")["test"]
    if dev:
        test_dataset = test_dataset.filter(lambda x: x["test_id"] in range(10))
    if task != "squad":
        tokenized_test = tokenize_data(task, test_dataset, tokenizer)
        tokenized_test = tokenized_test.remove_columns(
            [col for col in test_dataset.column_names if col != "id"])
    else:
        def preprocess_validation_examples(examples, max_length=384, stride=128):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs["offset_mapping"][i]
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                ]

            inputs["example_id"] = example_ids
            return inputs
        tokenized_test = test_dataset.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=test_dataset.column_names,
        )
        tokenized_test = tokenized_test.rename_column("example_id", "id")
    if return_train_val:
        val_dataset = load_from_disk(f"./data/{task}/{task}")["validation"]
        train_dataset = load_from_disk(f"./data/{task}/{task}")["train"]
        if task != "squad":
            tokenized_val = tokenize_data(task, val_dataset, tokenizer)
            tokenized_val = tokenized_val.remove_columns(
                [col for col in val_dataset.column_names if col != "id"])
            tokenized_train = tokenize_data(task, train_dataset, tokenizer)
            tokenized_train = tokenized_train.remove_columns(
                [col for col in train_dataset.column_names if col != "id"])
        else:
            tokenized_val = val_dataset.map(
                preprocess_validation_examples,
                batched=True,
                remove_columns=val_dataset.column_names,
            )
            tokenized_val = tokenized_val.rename_column("example_id", "id")
            tokenized_train = train_dataset.map(
                preprocess_validation_examples,
                batched=True,
                remove_columns=train_dataset.column_names,
            )
            tokenized_train = tokenized_train.rename_column("example_id", "id")
    if return_train_val:
        return (tokenized_train, tokenized_val, tokenized_test) if task != "squad" else ((tokenized_train, train_dataset), (tokenized_val, val_dataset), (tokenized_test, test_dataset))
    return tokenized_test if task != "squad" else (tokenized_test, test_dataset)


def get_preds(task, model, model_path, test_dataset, tokenizer, suite=False, get_metrics=False):

    if task == "sa":
        if not suite:
            model.config.problem_type = "single_label_classification"
            metric = load_metric("glue", "sst2")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)
        else:
            model.config.problem_type = "multi_label_classification"
        bs = 32
    elif task == "qqp":
        if not suite:
            metric = load_metric("glue", "qqp")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)
        bs = 32
    elif task == "squad":
        test_dataset, raw_dataset = test_dataset
        compute_metrics = get_squad_preds("mft")
        bs = 6

    training_args = TrainingArguments(output_dir=model_path,
                                      per_device_eval_batch_size=bs,
                                      seed=42)
    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer
    )

    preds, labels, _ = trainer.predict(test_dataset)

    if task == "squad":
        start_logits, end_logits = preds
        metrics, preds, labels = compute_metrics(
            start_logits, end_logits, test_dataset, raw_dataset, return_answers=True)
        if not suite:
            print(metrics)
    else:
        if not suite:
            metrics = compute_metrics((preds, labels))
            print(metrics)
    return (preds, labels, metrics) if get_metrics else (preds, labels)


def set_hyperparams(trainer, task, method, plus_iid=False, from_iid=False, args_path=None):
    if args_path is not None and args_path.is_file():
        print(f"loading hyperparams from {args_path}")
        args = torch.load(args_path)
        trainer.args.num_train_epochs = args.num_train_epochs
        trainer.args.per_device_train_batch_size = args.per_device_train_batch_size
        trainer.args.per_device_eval_batch_size = args.per_device_train_batch_size * 2
        trainer.args.learning_rate = args.learning_rate
        if method == "lp-ft":
            trainer.args.num_train_epochs = 3/2
        return
    print("loading hardset args")
    if task == "sa":
        if method == "default":
            if plus_iid:
                num_train_epochs = 2
                bs = 16
                learning_rate = 2e-05
            else:
                num_train_epochs = 1
                bs = 8
                learning_rate = 2e-05
        elif method == "dropout":
            if plus_iid:
                num_train_epochs = 2
                bs = 8
                learning_rate = 3e-05
            else:
                num_train_epochs = 1
                bs = 8
                learning_rate = 2e-05
        elif method == "l2":
            num_train_epochs = 1
            bs = 8
            learning_rate = 2e-05
            if plus_iid:
                num_train_epochs = 2
                learning_rate = 5e-05
        elif method in ["freeze", "lp-ft"]:
            num_train_epochs = 3
            bs = 8
            learning_rate = 1e-03
        elif method == "irm":
            num_train_epochs = 1
            bs = 8
            learning_rate = 5e-05
        elif method == "dro":
            num_train_epochs = 1
            bs = 8
            learning_rate = 2e-05
        elif method == "fish":
            num_train_epochs = 2
            bs = 8
            learning_rate = 2e-05
        else:
            raise NotImplementedError
    elif task == "qqp":
        if method == "default":
            num_train_epochs = 3
            bs = 16
            if plus_iid:
                learning_rate = 3e-05
            else:
                learning_rate = 2e-05
        elif method == "l2":
            num_train_epochs = 2
            bs = 8
            learning_rate = 2e-05
            if plus_iid:
                num_train_epochs = 3
                bs = 16
                learning_rate = 3e-05
        elif method == "dropout":
            num_train_epochs = 2
            bs = 8
            learning_rate = 2e-05
            if plus_iid:
                num_train_epochs = 3
                bs = 8
                learning_rate = 2e-05
        elif method in ["freeze", "lp-ft"]:
            num_train_epochs = 3
            bs = 8
            learning_rate = 1e-03
            if plus_iid:
                bs = 16
                if from_iid:
                    num_train_epochs = 1
                    bs = 8
        elif method == "irm":
            num_train_epochs = 3
            bs = 8
            learning_rate = 5e-05
        elif method == "dro":
            num_train_epochs = 3
            bs = 16
            learning_rate = 5e-05
        elif method == "fish":
            num_train_epochs = 3
            bs = 8
            learning_rate = 5e-05
        else:
            raise NotImplementedError
    elif task == "squad":
        if method == "default":
            num_train_epochs = 2
            bs = 3
            learning_rate = 3e-05
            if plus_iid:
                learning_rate = 2e-05
        elif method in ["freeze", "lp-ft"]:
            num_train_epochs = 3
            bs = 3
            learning_rate = 1e-03
            if plus_iid:
                if not from_iid:
                    num_train_epochs = 2
                    learning_rate = 5e-04
        elif method in ["l2", "dropout"]:
            num_train_epochs = 3
            bs = 2
            learning_rate = 2e-05
            if plus_iid and method == "l2":
                num_train_epochs = 1
                bs = 3
        elif method == "irm":
            num_train_epochs = 2
            bs = 3
            learning_rate = 2e-05
        elif method == "dro":
            num_train_epochs = 2
            bs = 2
            learning_rate = 2e-05
        else:
            raise NotImplementedError
    if method == "lp-ft":
        num_train_epochs = 3/2
    trainer.args.num_train_epochs = num_train_epochs
    trainer.args.per_device_train_batch_size = bs
    trainer.args.per_device_eval_batch_size = bs*2
    trainer.args.learning_rate = learning_rate


def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    """
    Measure and return speed performance metrics.
    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.
    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    return result


def ood_eval(task, model_path, tokenizer):
    datasets = []
    if task == "sa":
        datasets.append(load_dataset("imdb")["test"])
        datasets.append(load_dataset("yelp_review_full")["test"])

        tokenized_datasets = [tokenize_data(task, x.rename_column(
            "text", "test_case"), tokenizer) for x in datasets]

        def process_labels(examples):
            labels = []
            for label in examples['label']:
                if label in [0, 1]:
                    labels.append([1., 0.])
                elif label == 2:
                    labels.append([.5, .5])
                else:
                    labels.append([0., 1.])
            return {'label': labels}

        tokenized_datasets[1] = tokenized_datasets[1].map(
            process_labels, batched=True)
        metric = load_metric("glue", "sst2")

        def compute_metrics_imdb(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        def compute_metrics_yelp(eval_pred):
            logits, labels = eval_pred
            preds = softmax(logits, axis=-1)
            preds = squash_to_neutral(preds)
            labels = squash_to_neutral(labels)
            n = 0
            c = 0
            for p, l in zip(preds, labels):
                if p == int(l):
                    c += 1
                n += 1
            return {"accuracy": c/n}
        compute_metrics_funcs = [compute_metrics_imdb, compute_metrics_yelp]
        names = ["imdb", "yelp"]
    elif task == "qqp":
        datasets.append(load_dataset("paws", "labeled_final")["test"])
        qqp_paws = pd.read_csv("qqp-paws/dev_and_test.tsv", sep="\t")
        datasets.append(Dataset.from_pandas(qqp_paws))
        tokenized_datasets = [x.map(lambda e: tokenizer(e["sentence1"], e["sentence2"],
                                                        truncation=True, padding=True)) for x in datasets]
        metric = load_metric("glue", "qqp")
        compute_metrics = get_compute_metric(task, "mft")
        compute_metrics_funcs = 2 * [compute_metrics]
        names = ["paws-wiki", "paws-qqp"]
    elif task == "squad":
        datasets = [load_dataset("adversarial_qa", x, split="validation") for x in [
            "dbidaf", "dbert", "droberta"]]

        def preprocess_validation_examples(examples, max_length=384, stride=128):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs["offset_mapping"][i]
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                ]

            inputs["example_id"] = example_ids
            return inputs
        tokenized_datasets = [x.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=x.column_names) for x in datasets]
        metric = load_metric("squad")

        def compute_metrics(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30):
            example_to_features = collections.defaultdict(list)
            for idx, feature in enumerate(features):
                example_to_features[feature["example_id"]].append(idx)

            predicted_answers = []
            for example in tqdm(examples):
                example_id = example["id"]
                context = example["context"]
                answers = []

                # Loop through all features associated with that example
                for feature_index in example_to_features[example_id]:
                    start_logit = start_logits[feature_index]
                    end_logit = end_logits[feature_index]
                    offsets = features[feature_index]["offset_mapping"]

                    start_indexes = np.argsort(
                        start_logit)[-1: -n_best - 1: -1].tolist()
                    end_indexes = np.argsort(
                        end_logit)[-1: -n_best - 1: -1].tolist()
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            # Skip answers that are not fully in the context
                            if not offsets[start_index] or not offsets[end_index]:
                                continue
                            # Skip answers with a length that is either < 0 or > max_answer_length
                            if (
                                end_index < start_index
                                or end_index - start_index + 1 > max_answer_length
                            ):
                                continue
                            answer = {
                                "text": context[offsets[start_index][0]: offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                            answers.append(answer)

                # Select the answer with the best score
                if len(answers) > 0:
                    best_answer = max(answers, key=lambda x: x["logit_score"])
                    predicted_answers.append(
                        {"id": example_id,
                            "prediction_text": best_answer["text"]}
                    )
                else:
                    predicted_answers.append(
                        {"id": example_id, "prediction_text": ""})

            theoretical_answers = [
                {"id": ex["id"], "answers": ex["answers"]} for ex in examples]
            return metric.compute(predictions=predicted_answers, references=theoretical_answers), predicted_answers, theoretical_answers
        names = ["bidaf", "bert", "roberta"]

    training_args = TrainingArguments(output_dir=model_path,
                                      per_device_eval_batch_size=16,
                                      seed=42)
    for i, test_dataset in enumerate(tokenized_datasets):
        if task == "sa":
            if i == 0:
                model = BertForSequenceClassification.from_pretrained(
                    model_path, problem_type="single_label_classification")
            else:
                model = BertForSequenceClassification.from_pretrained(
                    model_path, problem_type="multi_label_classification")
        elif task == "qqp":
            model = BertForSequenceClassification.from_pretrained(model_path)
        elif task == "squad":
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        trainer = Trainer(
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics_funcs[i] if task != "squad" else None,
            model=model,
            tokenizer=tokenizer
        )
        print(f"Evaluating  {model_path} on dataset {names[i]}")
        if task != "squad":
            print(trainer.evaluate())
        else:
            predictions, _, _ = trainer.predict(test_dataset)
            start_logits, end_logits = predictions
            metrics, _, _ = compute_metrics(
                start_logits, end_logits, test_dataset, datasets[i])
            print(metrics)


class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly.
    From the original FISH implementation @ "https://github.com/YugeTen/fish/blob/333efa24572d99da0a4107ab9cc4af93a915d2a9/src/utils.py"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def fish_step(meta_weights, inner_weights, meta_lr):
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)
    return meta_weights
