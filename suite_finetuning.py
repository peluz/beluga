from datasets import load_from_disk, concatenate_datasets

from pathlib import Path
import torch
import argparse
import pickle
from behaviour_training import *
from utils import initialize_seeds, pred_and_conf
from ray import tune


def main(task, split="all", dev=False, method="default", plus_iid=False, from_iid=False):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    initialize_seeds()
    np.set_printoptions(
            formatter={lambda x: "{0:0.5f}".format(x)}, suppress=True)

    if task == "sa":
        task_data = "SST2"
    else:
        task_data = task
    if plus_iid:
        if from_iid:
            MODEL_PATH = Path(
                f"./models/BERT-{task_data}-iid-all+iid-{method}").absolute()
            PRETRAINED_PATH = Path(f"./models/BERT-{task}").absolute()
        else:
            MODEL_PATH = Path(
                f"./models/BERT-{task_data}-all+iid-{method}").absolute()
            if task != "squad":
                PRETRAINED_PATH = 'bert-base-uncased'
            else:
                if dev:
                    PRETRAINED_PATH = "distilbert-base-uncased"
                else:
                    PRETRAINED_PATH = "bert-large-uncased-whole-word-masking"
    elif method == "default":
        MODEL_PATH = Path(
            f"./models/BERT-{task_data}-all").absolute()
        PRETRAINED_PATH = Path(f"./models/BERT-{task}").absolute()
    else:
        MODEL_PATH = Path(
            f"./models/BERT-{task_data}-all-{method}").absolute()
        PRETRAINED_PATH = Path(f"./models/BERT-{task}").absolute()
    DATA_PATH = Path(f"./data/{task}/{task}/").absolute()

    train_dataset, eval_dataset, tokenizer, id_to_samples_inv, id_to_samples_dir, soft_to_label, inv_datasets, clean_inv_datasets, perturb_inv_datasets = process_data(
        task, DATA_PATH, plus_iid, dev)

    logging_steps = 500 
    if method == "irm":
        logging_steps /= 50
    elif method == "fish":
        logging_steps /= 5
    bs = 2 if task == "squad" else 8

    args = {
        "output_dir": MODEL_PATH,
        "evaluation_strategy": "epoch",
        "logging_steps": logging_steps,
        "disable_tqdm": True,
        "learning_rate": 1e-5,
        "per_device_eval_batch_size": bs*2,
        "per_device_train_batch_size": bs,
        "save_strategy": "no",
        "seed": 42,
        "remove_unused_columns": False,
        "num_train_epochs": 1,
    }

    training_args = TrainingArguments(**args)
    jump_all = Path(MODEL_PATH/"pytorch_model.bin").exists()
    print(f"{MODEL_PATH} exists.")

    if split in ["all", "full_run"] and not jump_all:
        # perform hyperparameter tuning using grid search
        if Path(MODEL_PATH/"pytorch_model.bin").exists():
            print(f"Jumping completed model {MODEL_PATH}")
            pass

        def custom_hp_space(trial):
            if task == "squad":
                bs = [2, 3]
            else:
                bs = [8, 16]
            if method == "freeze":
                lrs = [1e-4, 5e-4, 1e-3]
            else:
                lrs = [2e-5, 3e-5, 5e-5]
            space = {
                "learning_rate": tune.grid_search(lrs),
                "per_device_train_batch_size": tune.grid_search(bs),
                "num_train_epochs": tune.grid_search([1, 2, 3])
            }
            
            if dev:
                space["learning_rate"]= tune.grid_search([2e-5])
                space["num_train_epochs"]= tune.grid_search([1])
                
            return space

        if method != "lp-ft":
            trainer = get_trainer(task, tokenizer, training_args, train_dataset, eval_dataset,
                                  id_to_samples_inv, id_to_samples_dir, soft_to_label, PRETRAINED_PATH, plus_iid, dev, method)
            best_run = trainer.hyperparameter_search(
                backend='ray',
                hp_space=custom_hp_space,
                direction='maximize',
                n_trials=1,
                compute_objective=get_compute_objective(task),
                progress_reporter=tune.CLIReporter(max_report_frequency=60),
                fail_fast=True,
                local_dir="./ray_results",
                resources_per_trial={
                                "cpu": 8,
                                "gpu": 1
                            },
            )

            print(best_run)

            for n, v in best_run.hyperparameters.items():
                setattr(trainer.args, n, v)
        else:
            trainer = get_trainer(task, tokenizer, training_args, train_dataset, eval_dataset,
                                  id_to_samples_inv, id_to_samples_dir, soft_to_label, PRETRAINED_PATH, plus_iid, dev, method)
            set_hyperparams(trainer, task, method, plus_iid, from_iid, args_path=Path(str(MODEL_PATH).replace("lp-ft", "freeze")))

        trainer.train()

        if method == "lp-ft":
            if plus_iid:
                args_path = Path(str(MODEL_PATH).replace("lp-ft", "default"))/"training_args.bin"
            else:
                args_path = Path(str(MODEL_PATH).replace("-lp-ft", ""))/"training_args.bin"
            set_hyperparams(trainer, task, "default", plus_iid, from_iid, args_path=args_path)
            trainer.args.num_train_epochs = 3/2
            for param in trainer.model.base_model.parameters():
                param.requires_grad = True
            # Use the linear probe-trained model but reset optimizer state
            trainer.model_init = lambda: trainer.model
            trainer.train()

        if not dev:
            trainer.save_model(MODEL_PATH)

        if method == "dro":
            print("Obtained group weights:")
            for i, k in enumerate(trainer.group2idx.keys()):
                print(f"{k}: {trainer.group_weights[i]}")

        iid_test = get_iid_test(task, tokenizer, dev)
        suite_test = get_suite_test(task, tokenizer, dev)

        suite_path = Path(f"./data/{task}/predictions/checklist/")
        suite_path.mkdir(exist_ok=True, parents=True)
        if plus_iid:
            pred_file = suite_path/f"BERT-{task_data}-all+iid-{method}"
            if from_iid:
                pred_file = suite_path/f"BERT-{task_data}-iid-all+iid-{method}"
        elif method == "default":
            pred_file = suite_path/f"BERT-{task_data}-all"
        else:
            pred_file = suite_path/f"BERT-{task_data}-all-{method}"
        _, _ = get_preds(task, trainer.model, MODEL_PATH, iid_test, tokenizer)
        preds, labels = get_preds(task, trainer.model, MODEL_PATH, suite_test, tokenizer, suite=True)
        if not dev:
            if task == "sa":
                preds = softmax(preds, axis=-1)
                preds_tri, pp = pred_and_conf(preds)
                with open(pred_file, "w") as file:
                    for pred, p in zip(preds_tri, pp):
                        file.write(f"{pred} {p[0]} {p[1]} {p[2]}\n")
            elif task == "qqp":
                preds = softmax(preds, axis=-1)
                with open(pred_file, "w") as file:
                    for pred in preds:
                        file.write(f"{pred[1]}\n")
            elif task == "squad":
                with open(pred_file, "w") as file:
                    for answer in preds:
                        answer = answer["prediction_text"].replace('\n', ' ')
                        file.write(f'{answer}\n')
            ood_eval(task, MODEL_PATH, tokenizer)

    if split != "all":
        if split == "full_run":
            splits = ["aspectOut", "classOut", "funcOut"]
        else:
            splits = [split]

        iid_test = get_iid_test(task, tokenizer, dev)
        suite_test = get_suite_test(task, tokenizer, dev)

        for s in splits:
            if plus_iid:
                if from_iid:
                    iid_path = Path(
                        f"./data/{task}/predictions/target/iid-{s}+iid-{method}")
                    suite_path = Path(
                        f"./data/{task}/predictions/checklist/iid-{s}+iid-{method}")
                else:
                    iid_path = Path(
                        f"./data/{task}/predictions/target/{s}+iid-{method}")
                    suite_path = Path(
                        f"./data/{task}/predictions/checklist/{s}+iid-{method}")
            else:
                iid_path = Path(
                    f"./data/{task}/predictions/target/{s}-{method}")
                suite_path = Path(
                    f"./data/{task}/predictions/checklist/{s}-{method}")
            iid_path.mkdir(exist_ok=True, parents=True)
            suite_path.mkdir(exist_ok=True, parents=True)

            iterable = set()
            if s =="aspectOut":
                iterable.update(train_dataset.keys())
                if plus_iid:
                    iterable.remove("iid")
            if s =="classOut":
                for k, v in train_dataset.items():
                    if k != "iid":
                        iterable.update(v["capability"])
            if s =="funcOut" :
                for k, v in train_dataset.items():
                    if k != "iid":
                        iterable.update(v["functionality"])

            for out in iterable:
                if s == "funcOut":
                    filtered_train = {k: v.filter(
                        lambda x: x["functionality"] != out) if k != "iid" else v for k, v in train_dataset.items()}
                elif s == "classOut":
                    filtered_train = {k: v.filter(
                        lambda x: x["capability"] != out) if k != "iid" else v for k, v in train_dataset.items()}
                elif s == "aspectOut":
                    filtered_train = {k: v for k,
                                    v in train_dataset.items() if k != out}

                iid_file = iid_path/f"{out}Out".replace("/", "|")
                suite_file = suite_path/f"{out}Out".replace("/", "|")
                if Path(iid_file).exists() and Path(suite_file).exists():
                    print(f"Jumping completed model {out}Out")
                    continue

                print()
                print(f"Training without {out}")
                print()
                trainer = get_trainer(task, tokenizer, training_args, filtered_train, eval_dataset,
                                    id_to_samples_inv, id_to_samples_dir, soft_to_label, PRETRAINED_PATH, plus_iid, dev, method)
                args_path = MODEL_PATH/"training_args.bin"
                print(args_path)
                if method == "lp-ft":
                    args_path = Path(str(args_path).replace(method, "freeze"))                    
                set_hyperparams(trainer, task, method, plus_iid, from_iid, args_path=args_path)
                trainer.train()
                if method == "lp-ft":
                    print()
                    print("Fine-tuning after linear probing")
                    print()
                    if plus_iid:
                        args_path = Path(str(args_path).replace("freeze", "default"))
                    else:
                        args_path = Path(str(args_path).replace("-freeze", ""))
                    set_hyperparams(trainer, task, "default", plus_iid, from_iid, args_path=args_path)
                    trainer.args.num_train_epochs = 3/2
                    for param in trainer.model.base_model.parameters():
                        param.requires_grad = True
                    # Use the linear probe-trained model but reset optimizer state
                    trainer.model_init = lambda: trainer.model
                    trainer.train()
                # Save iid_data_preds
                print()
                print("Evaluating on iid test data")
                print()
                preds, labels = get_preds(
                    task, trainer.model, MODEL_PATH, iid_test, tokenizer)
                if not dev:
                    with open(iid_file, "wb") as file:
                        pickle.dump({"preds": preds, "labels": labels}, file)
                # Save checklist preds, check each notebook
                print()
                print("Generating test suite preds")
                print()
                preds, labels = get_preds(
                    task, trainer.model, MODEL_PATH, suite_test, tokenizer, suite=True)
                if not dev:
                    if task == "sa":
                        preds = softmax(preds, axis=-1)
                        preds_tri, pp = pred_and_conf(preds)
                        with open(suite_file, "w") as file:
                            for pred, p in zip(preds_tri, pp):
                                file.write(f"{pred} {p[0]} {p[1]} {p[2]}\n")
                    elif task == "qqp":
                        preds = softmax(preds, axis=-1)
                        with open(suite_file, "w") as file:
                            for pred in preds:
                                file.write(f"{pred[1]}\n")
                    elif task == "squad":
                        with open(suite_file, "w") as file:
                            for answer in preds:
                                answer = answer["prediction_text"].replace(
                                    '\n', ' ')
                                file.write(f'{answer}\n')
                print(f"Finished training without {out}")
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune a model on test suite data')
    parser.add_argument('task', help='The training task.',
                        type=str, choices=["sa", "qqp", "squad"])
    parser.add_argument('--split', help='The test suite splitting method.', type=str,
                        choices=["all", "funcOut", "classOut", "aspectOut", "full_run"], default="full_run")
    parser.add_argument('--method', help='The training method.', type=str, choices=[
                        "default", "l2", "dropout", "freeze", "lp-ft", "irm", "dro", "fish"], default="default")
    parser.add_argument('--plus_iid', help='Include iid data for training.',
                        type=str, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--from_iid', help='When including iid data for training, start from IID model',
                        type=str, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dev', help='If testing.', type=str,
                        action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args.task, split=args.split, dev=args.dev, method=args.method,
         plus_iid=args.plus_iid, from_iid=args.from_iid)
