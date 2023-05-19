#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.stats import binomtest
from pathlib import Path
from behaviour_training import *
from transformers import BertForSequenceClassification, BertForQuestionAnswering, AutoTokenizer
from scipy.stats import hmean
import random
from tqdm.auto import tqdm
import pickle
import pandas as pd
from copy import deepcopy
import string
import re


# In[2]:


tasks = ["sa", "qqp", "squad"]
configs = ["iid-t", "iid+t", "iid-iid+t"]
methods = ["default", "l2", "dropout", "freeze", "lp-ft", "irm", "dro", "fish"]


# ## Generate target preds if needed

# In[3]:


def get_model_name(task, config, method, baseline=False):
    if task == "sa":
        task_data = "SST2"
    else:
        task_data = task
    if baseline:
        return f"BERT-{task_data}"
    if method == "default":
        suffix = ""
    else:
        suffix = f"-{method}"
    if config == "iid-t":
        return f"BERT-{task_data}-all{suffix}"
    elif config == "iid+t":
        return f"BERT-{task_data}-all+iid-{method}"
    else:
        return f"BERT-{task_data}-iid-all+iid-{method}"


# In[5]:


tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased')


# In[6]:


iid_tests = {x: get_iid_test(x, tokenizer) for x in tasks}


# In[7]:


all_preds = {}


# In[8]:


for task in tasks:
    model_name = get_model_name(task, None, None, baseline=True)
    baseline_path = Path(f"./data/{task}/predictions/target/{model_name}").absolute()
    try:
        with open(baseline_path, "rb") as file:
            preds = pickle.load(file)
            all_preds.setdefault(task, {})["baseline"] = preds
    except FileNotFoundError as e:
        model_path = Path(f"./models/{model_name}").absolute()
        if task != "squad":
            model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            model = BertForQuestionAnswering.from_pretrained(model_path)
        preds, _ = get_preds(task, model, model_path, iid_tests[task], tokenizer)
        all_preds.setdefault(task, {})["baseline"] = preds
        with open(baseline_path, "wb") as file:
            pickle.dump(preds, file)


# In[10]:


for task in tasks:
    for config in configs:
        for method in methods:
            model_name = get_model_name(task, config, method)
            results_path = Path(f"./data/{task}/predictions/target/{model_name}").absolute()
            try:
                with open(results_path, "rb") as file:
                    preds = pickle.load(file)
                    all_preds.setdefault(task, {})[(config, method)] = preds
            except FileNotFoundError as e:
                model_path = Path(f"./models/{model_name}").absolute()
                if task != "squad":
                    model = BertForSequenceClassification.from_pretrained(model_path)
                else:
                    model = BertForQuestionAnswering.from_pretrained(model_path)
                preds, _ = get_preds(task, model, model_path, iid_tests[task], tokenizer)
                all_preds.setdefault(task, {})[(config, method)] = preds
                with open(results_path, "wb") as file:
                    pickle.dump(preds, file)


# In[11]:


all_preds["sa"].keys()


# ### Dataset results significance testing

# In[12]:


# SQuAD evaluation script from https://github.com/huggingface/datasets/

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# In[13]:


def get_significance(task, preds_1, preds_2, labels):
    if task != "squad":
        preds_1 = preds_1.argmax(-1)
        preds_2 = preds_2.argmax(-1)
        corrects1 = (np.array(preds_1) == np.array(labels)).astype(int)
        corrects2 = (np.array(preds_2) == np.array(labels)).astype(int)
    else:
        corrects1 = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds_1, labels)]).astype(int)
        corrects2 = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds_2, labels)]).astype(int)
    diffs = corrects2 - corrects1
    successes = (diffs == 1).sum()
    trials = np.abs(diffs).sum()
    return binomtest(successes, trials)


# In[16]:


try:
    with open("./data/dataset_pvalues", "rb") as file:
        pvalues = pickle.load(file)
except FileNotFoundError:
    pvalues = {}
    for task in tasks:
        if task != "squad":
            labels = iid_tests[task]["label"]
        else:
            labels =[x["text"] for x in iid_tests["squad"][1]["answers"]]
        for config in configs:
            for method in methods:
                print(f"Comparing baseline and {config}+{method} for task {task}")
                sig_test = get_significance(task, all_preds[task]["baseline"], all_preds[task][(config, method)], labels)
                print(sig_test)
                if sig_test.pvalue > .05:
                    print("Difference is not significant")
                print()
                pvalues.setdefault(task, {})[(config, method)] = sig_test.pvalue
    with open("./data/dataset_pvalues", "wb") as file:
        pickle.dump(pvalues, file)


# In[21]:


df = pd.DataFrame.from_dict(pvalues)


# In[23]:


df[df >.05]


# ## Generalisation scores significance testing

# In[24]:


with open("./data/suite_hits.pkl", "rb") as file:
    all_hits = pickle.load(file)


# In[25]:


# def randomized_test(hits_dataset1, hits_suite1, hits_dataset2, hits_suite2, trials, seed=42, verbose=False):
#     suite_score1 = np.mean([np.nanmean(v) for v in hits_suite1.values()])
#     suite_score2 = np.mean([np.nanmean(v) for v in hits_suite2.values()])
#     dataset_score1 = np.nanmean(hits_dataset1)
#     dataset_score2 = np.nanmean(hits_dataset2)
#     score1 = hmean([suite_score1, dataset_score1])
#     score2 = hmean([suite_score2, dataset_score2])
#     print('# score(model1) = %f' % score1)
#     print('# score(model2) = %f' % score2)

#     diff = abs(score1 - score2)
#     print('# abs(diff) = %f' % diff)
#     if verbose:
#         print()
#         print(f""""Baseline dataset: {dataset_score1},
#                       Baseline suite and avg: {[(k, np.nanmean(v)) for k, v in hits_suite1.items()]} {suite_score1}
#                       Gscore {score1}""")
#         print(f""""method dataset: {dataset_score2},
#               method suite and avg: {[(k, np.nanmean(v)) for k, v in hits_suite2.items()]} {suite_score2}
#               Gscore {score2}""")
#         print()


#     uncommon_dataset = [i for i in range(len(hits_dataset1)) if hits_dataset1[i] != hits_dataset2[i]]
#     uncommon_suite = {k: [i for i in range(len(v)) if v[i] != hits_suite2[k][i]] for k, v in hits_suite1.items()}

#     better = 0
    
#     rng = random.Random(seed)
#     getrandbits_func = rng.getrandbits

#     for _ in tqdm(range(trials)):
#         dataset1, dataset2 = list(hits_dataset1), list(hits_dataset2)
#         suite1, suite2 = deepcopy(hits_suite1), deepcopy(hits_suite2)
#         for i in uncommon_dataset:
#             if getrandbits_func(1) == 1:
#                 dataset1[i], dataset2[i] = hits_dataset2[i], hits_dataset1[i]
#         assert len(hits_dataset1) == len(hits_dataset2) == len(dataset1) == len(dataset2)
        
#         for k in hits_suite1.keys():
#             for i in uncommon_suite[k]:
#                 if getrandbits_func(1) == 1:
#                     suite1[k][i], suite2[k][i] = hits_suite2[k][i], hits_suite1[k][i]
#             assert len(suite1[k]) == len(suite2[k]) == len(hits_suite2[k]) == len(hits_suite1[k])
        
#         new_suite_score1 = np.mean([np.nanmean(v) for v in suite1.values()])
#         new_suite_score2 = np.mean([np.nanmean(v) for v in suite2.values()])
#         new_dataset_score1 = np.nanmean(dataset1)
#         new_dataset_score2 = np.nanmean(dataset2)
#         new_score1 = hmean([new_suite_score1, new_dataset_score1])
#         new_score2 = hmean([new_suite_score2, new_dataset_score2])
#         diff_local = abs(new_score1 - new_score2)

#         if diff_local >= diff:
#             better += 1
#             if verbose:
#                 print(diff_local)
#                 print(f""""delta dataset: {new_dataset_score1 - dataset_score1},
#                       delta suite and avg: {[(k, np.nanmean(v) - np.nanmean(hits_suite1[k])) for k, v in suite1.items()]} {new_suite_score1 - suite_score1}
#                       Gscore {new_score1 - score1}""")
#                 print(f""""delta dataset: {new_dataset_score2 - dataset_score2},
#                       delta suite and avg: {[(k, np.nanmean(v) - np.nanmean(hits_suite2[k])) for k, v in suite2.items()]} {new_suite_score2 - suite_score2}
#                       Gscore {new_score2 - score2}""")
#                 print()

#     p = (better + 1) / (trials + 1)
#     print(f"p_value: {p}, successes: {better}")
#     return p


# # In[26]:


# scores = ["seen", "funcOut", "classOut", "aspectOut"]


# # In[248]:


# try:
#     with open("./data/suite_pvalues", "rb") as file:
#         pvalues = pickle.load(file)
# except FileNotFoundError:
#     pvalues = {}
#     for task in tasks:
#         if task != "squad":
#             labels = iid_tests[task]["label"]
#             baseline_dataset_hits = (all_preds[task]["baseline"].argmax(-1) == labels).astype(int)
#         else:
#             labels =[x["text"] for x in iid_tests["squad"][1]["answers"]]
#             baseline_dataset_hits = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(all_preds[task]["baseline"], labels)]).astype(int)
#         baseline_suite_hits = all_hits[task]["baseline"]
#         for config in configs:
#             for method in methods:
#                 for score in scores:
#                     print(f"Comparing baseline and {config}+{method} for task {task} and score {score}")
#                     if task != "squad":
#                         method_dataset_hits = (all_preds[task][(config, method)].argmax(-1) == labels).astype(int)
#                     else:
#                         method_dataset_hits = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(all_preds[task][(config, method)], labels)]).astype(int)
#                     method_suite_hits = all_hits[task][(config, method)][score]
#                     sig_test = randomized_test(baseline_dataset_hits, baseline_suite_hits,
#                                                method_dataset_hits, method_suite_hits, 10000, seed=42)
#                     pvalues.setdefault(task, {}).setdefault((config, method), {})[score] = sig_test
#                     if sig_test > .05:
#                         print("Difference is not significant")
#                     print()
#     with open("./data/suite_pvalues", "wb") as file:
#         pickle.dump(pvalues, file)



# In[74]:


def get_score(hits_dataset_by_task, hits_suite_by_task_score):
    gscores = {}
    scores = ["seen", "funcOut", "classOut", "aspectOut"]
    for task, hits_dataset in hits_dataset_by_task.items():
        dataset_score = np.nanmean(hits_dataset)
        suite_scores = [np.mean([np.nanmean(v) for v in hits_suite.values()]) for hits_suite in hits_suite_by_task_score[task]]
        gscores[task] = {gscore: hmean([dataset_score, suite_score]) for gscore, suite_score in zip(scores, suite_scores)}
    gscores["avg"] = np.mean([x for task_scores in gscores.values() for x in task_scores.values()])
    return gscores


# In[210]:


score_to_idx = {score: k for k, score in enumerate(["seen", "funcOut", "classOut", "aspectOut"])}

def randomized_test_avg(hits_dataset_by_task1, hits_suite_by_task_score1, hits_dataset_by_task2, hits_suite_by_task_score2, trials, seed=42, verbose=False):
    g_scores1 = get_score(hits_dataset_by_task1, hits_suite_by_task_score1)
    g_scores2 = get_score(hits_dataset_by_task2, hits_suite_by_task_score2)
    for task, g_scores in g_scores1.items():
        if task == "avg":
            print('# avg score(model1) = %f' % g_scores)
            print()
        else:
            for score, value in g_scores.items():
                    print(f"{score} score(model 1) for task {task} = {value}")
    for task, g_scores in g_scores2.items():
        if task == "avg":
            print('# avg score(model2) = %f' % g_scores)
            print()
        else:
            for score, value in g_scores.items():
                    print(f"{score} score(model 2) for task {task} = {value}")
    
    diffs = {}
    for task, values in g_scores1.items():
        if task == "avg":
            diffs["avg"] = abs(values - g_scores2[task])
            print('# abs avg(diff) = %f' % diffs["avg"])
            print()
        else:
            for score, value in values.items():
                diffs.setdefault(task, {})[score] = abs(value - g_scores2[task][score])
                print(f"abs(diff) for {task} {score} = {diffs[task][score]}")

    uncommon_datasets = {k: [i for i in range(len(v)) if v[i] != hits_dataset_by_task2[k][i]] for k, v in hits_dataset_by_task1.items()}
    uncommon_suites = {task: [{k: [i for i in range(len(v)) if v[i] != hits_suite_by_task_score2[task][n][k][i]] for k, v in result.items()}
                              for n, result in enumerate(results)] for task, results in hits_suite_by_task_score1.items()}
    better = {}
    for task, diff in diffs.items():
        if task == "avg":
            better["avg"] = 0
        else:
            for score, value in diff.items():
                better.setdefault(task, {})[score] = 0
    
    rng = random.Random(seed)
    getrandbits_func = rng.getrandbits

    for _ in tqdm(range(trials)):
        datasets1, datasets2 = deepcopy(hits_dataset_by_task1), deepcopy(hits_dataset_by_task2)
        suite1, suite2 = deepcopy(hits_suite_by_task_score1), deepcopy(hits_suite_by_task_score2)
        for task, uncommons in uncommon_datasets.items():
            for i in uncommons:
                if getrandbits_func(1) == 1:
                    datasets1[task][i], datasets2[task][i] = hits_dataset_by_task2[task][i], hits_dataset_by_task1[task][i]
        
        for task, scores in uncommon_suites.items():
            for n, funcs in enumerate(scores):
                for k, uncommons in funcs.items():
                    for i in uncommons:
                        if getrandbits_func(1) == 1:
                            suite1[task][n][k][i], suite2[task][n][k][i] = hits_suite_by_task_score2[task][n][k][i], hits_suite_by_task_score1[task][n][k][i]

        new_score1 = get_score(datasets1, suite1)
        new_score2 = get_score(datasets2, suite2)

        for task, values in new_score1.items():
            if task == "avg":
                if abs(values - new_score2[task]) >= diffs["avg"]:
                    better["avg"] += 1
                    if verbose:
                        print("New avg scores:")
                        print(values, new_score2[task])
            else:
                for score, value in values.items():
                    if abs(value - new_score2[task][score]) >= diffs[task][score]:
                        better[task][score] += 1
                        if verbose:
                            print(f"delta_score for {task} {score}")
                            print(value - new_score2[task][score])
                            print("delta_dataset:")
                            print(np.mean(datasets1[task]) - np.mean(hits_dataset_by_task1[task]))
                            print(np.mean(datasets2[task]) - np.mean(hits_dataset_by_task2[task]))
                            print("delta_suite:")
                            print({k: np.nanmean(v) - np.nanmean(hits_suite_by_task_score1[task][score_to_idx[score]][k]) for k, v in suite1[task][score_to_idx[score]].items()})
                            print({k: np.nanmean(v) - np.nanmean(hits_suite_by_task_score2[task][score_to_idx[score]][k]) for k, v in suite2[task][score_to_idx[score]].items()})
    ps = {}                     
    for task, values in better.items():
        if task == "avg":
            ps["avg"] =  (values + 1) / (trials + 1)
            print(f"p_value: {ps['avg']}, successes: {better['avg']}, avg")
        else:
            for score, value in values.items():
                ps.setdefault(task, {})[score] = (value + 1) / (trials + 1)
                print(f"p_value: {ps[task][score]}, successes: {better[task][score]} for {task} {score}")
    print()
    return ps


# In[251]:


pvalues = {}
baseline_hits_dataset_by_task = {}
baseline_hits_suite_by_task_score = {}
labels_by_task = {}
for task in tasks:
    if task != "squad":
        labels = iid_tests[task]["label"]
        baseline_hits_dataset_by_task[task] = (all_preds[task]["baseline"].argmax(-1) == labels).astype(int)
    else:
        labels =[x["text"] for x in iid_tests["squad"][1]["answers"]]
        baseline_hits_dataset_by_task[task] = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(all_preds[task]["baseline"], labels)]).astype(int)
    baseline_hits_suite_by_task_score[task] = [deepcopy(all_hits[task]["baseline"]) for _ in range(4)]
    labels_by_task[task] = labels
for config in configs:
    for method in methods:
        print(f"Comparing baseline and {config}+{method}")
        method_hits_dataset_by_task = {}
        method_hits_suite_by_task_score = {}
        for task in tasks:
            labels = labels_by_task[task]
            if task != "squad":
                method_hits_dataset_by_task[task] = (all_preds[task][(config, method)].argmax(-1) == labels).astype(int)
            else:
                method_hits_dataset_by_task[task] = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(all_preds[task][(config, method)], labels)]).astype(int)
            method_hits_suite_by_task_score[task] = list(all_hits[task][(config, method)].values())
        sig_test = randomized_test_avg(baseline_hits_dataset_by_task, baseline_hits_suite_by_task_score,
                                   method_hits_dataset_by_task, method_hits_suite_by_task_score, 10000, seed=42)
        pvalues[(config, method)] = sig_test    


# In[231]:


with open("./data/pvalues_suite_avg", "wb") as file:
    pickle.dump(pvalues, file)

