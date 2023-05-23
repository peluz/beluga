# Cross-functional Analysis of Generalisation in Behavioural Learning

This repo holds source code for the paper ["Cross-functional Analysis of Generalisation in Behavioural Learning"](http://arxiv.org/abs/2305.12951), to be published in the Transactions of the Association for Computational Linguistics (TACL).

<!-- ## Citation

If you use the code or data from this repository, please consider citing the paper: -->
## Requirement

- [Anaconda](https://www.anaconda.com/download)

## Setting up

1. Run the snippet below to install all dependencies:

```console
conda env create -f environment.yml
```
2. Download the test suites from https://github.com/marcotcr/checklist/blob/master/release_data.tar.gz and extract the files to the 'data' folder.

## Generating the data
1. Run notebook 'checkList_to_csv.ipynb' to convert the suites to csv files.
2. Run notebook 'make_suite_dataset.ipynb' to convert the csv files to datasets with predefined splits.

## Reproducing the experiments
- Script 'iid_finetuning.py' trains the IID models
- Script 'suite_finetuning.py' trains the suite-augmented models. Usage:
    ```console
    python suite_finetuning.py {task} --method {method} {--plus_iid} {--from_iid}
    ```
    - {task} can be sa, qqp or squad. 
    - {method} can be default, l2, dropout, freeze, lp-ft, irm, dro or fish.
    - add --plus_iid for the IID+T configuration
    - add --plus_iid and --from_iid for the IID->IID+T configuration
    - Do not add neither for the IID->T configuration.
- Notebook 'suites_evaluation.ipynb' generates suite results.
- Script 'significance_testing.py' runs the significance tests.
- Notebook 'gen_graphics.ipynb' generates paper figures and tables.