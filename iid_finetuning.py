#!/usr/bin/env python
# coding: utf-8
from ray import tune
from datasets import load_dataset, load_metric, load_from_disk
from transformers import BertForSequenceClassification, TrainingArguments, BertTokenizer, BertTokenizerFast, Trainer, AutoTokenizer, AutoModelForQuestionAnswering
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import collections

from utils import initialize_seeds


for task in ["sst2", "qqp", "squad"]:

    if task == "sst2":
        MODEL_PATH = Path("./models/BERT-SST2")
    elif task == "qqp":
        MODEL_PATH = Path("./models/BERT-qqp")
    elif task == "squad":
        MODEL_PATH = Path("./models/BERT-squad")

    initialize_seeds()

    if task == "squad":
        dataset = load_dataset("squad")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking")
    else:
        dataset = load_dataset("glue", task)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_training_examples(examples, max_length=384, stride=128):
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

        offset_mapping = inputs.pop("offset_mapping")
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

    if task == "squad":
        try:
            train_dataset = load_from_disk(
                "./data/squad/squad_target_data/train")
            val_dataset = load_from_disk("./data/squad/squad_target_data/val")
            test_dataset = load_from_disk(
                "./data/squad/squad_target_data/test")
            val_dataset_raw = load_from_disk(
                "./data/squad/squad_target_data/val_raw")
            test_dataset_raw = load_from_disk(
                "./data/squad/squad_target_data/test_raw")
        except FileNotFoundError:
            train_dataset = dataset["train"].map(
                preprocess_training_examples,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            train_dataset.save_to_disk("./data/squad/squad_target_data/train")
            val_test_datasets = dataset["validation"].train_test_split(
                test_size=0.5, seed=42)
            val_dataset_raw = val_test_datasets["train"]
            test_dataset_raw = val_test_datasets["test"]

            val_dataset = val_dataset_raw.map(
                preprocess_validation_examples,
                batched=True,
                remove_columns=val_dataset_raw.column_names,
            )
            test_dataset = test_dataset_raw.map(
                preprocess_validation_examples,
                batched=True,
                remove_columns=test_dataset_raw.column_names,
            )
            val_dataset.save_to_disk("./data/squad/squad_target_data/val")
            test_dataset.save_to_disk("./data/squad/squad_target_data/test")
            val_dataset.save_to_disk("./data/squad/squad_target_data/val_raw")
            test_dataset.save_to_disk(
                "./data/squad/squad_target_data/test_raw")
    else:
        try:
            tokenized_datasets = dataset.load_from_disk(
                f"./data/{task}/{task}_target_data/")
        except FileNotFoundError:
            if task == "sst2":
                tokenized_datasets = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length",
                                                                     max_length=128, truncation=True))
            elif task == "qqp":
                tokenized_datasets = dataset.map(lambda e: tokenizer(e["question1"], e["question2"],
                                                                     truncation=True, padding=True))
            tokenized_datasets.save_to_disk(
                f"./data/{task}/{task}_target_data/")

            # Test set labels are private, so we split the validation set in half for validation and testing
        train_dataset = tokenized_datasets["train"]
        val_test_datasets = tokenized_datasets["validation"].train_test_split(
            test_size=0.5, seed=42)
        val_dataset = val_test_datasets["train"]
        test_dataset = val_test_datasets["test"]

    if task == "squad":
        model = AutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking")

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
            return metric.compute(predictions=predicted_answers, references=theoretical_answers)

        metric = load_metric("squad")

        training_args = TrainingArguments(output_dir=MODEL_PATH,
                                          evaluation_strategy="no",
                                          disable_tqdm=True,
                                          learning_rate=3e-5,
                                          num_train_epochs=2,
                                          per_device_eval_batch_size=3,
                                          per_device_train_batch_size=3,
                                          save_strategy="no",
                                          seed=42
                                          )
        trainer = Trainer(
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            model=model
        )
    else:
        def model_init():
            return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        metric = load_metric("glue", config_name=task)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(output_dir=MODEL_PATH,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=True,
                                          learning_rate=2e-5,
                                          per_device_eval_batch_size=16,
                                          per_device_train_batch_size=16,
                                          save_strategy="no",
                                          seed=42
                                          )
        trainer = Trainer(
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            model_init=model_init
        )

    trainer.train()
    trainer.save_model(MODEL_PATH)

    if task == "squad":
        predictions, _, _ = trainer.predict(val_dataset)
        start_logits, end_logits = predictions
        print(compute_metrics(start_logits, end_logits,
              val_dataset, val_dataset_raw))
        predictions, _, _ = trainer.predict(test_dataset)
        start_logits, end_logits = predictions
        print(compute_metrics(start_logits, end_logits,
              test_dataset, test_dataset_raw))
    else:
        trainer.evaluate()
        trainer.evaluate(test_dataset)
