import random
import torch
import numpy as np
from sklearn.metrics import classification_report

def initialize_seeds(seed=42):
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.detesrministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)
    
def get_results(trainer, dataset):
    results = trainer.predict(dataset)
    for metric in results.metrics:
        print(metric, results.metrics['{}'.format(metric)])
    preds=[]
    for row in results[0]:
        preds.append(int(np.argmax(row)))
    print(classification_report(dataset[:]["label"],preds,digits=4))
    return preds

def pred_and_conf(preds):
    # change format to softmax, make everything in [0.33, 0.66] range be predicted as neutral
    pr = preds[:,1]
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
    pp[neutral_pos, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_pos] - 0.5)
    pp[neutral_pos, 2] = 1 - pp[neutral_pos, 1]
    neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
    pp[neutral_neg, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_neg] - 0.5)
    pp[neutral_neg, 0] = 1 - pp[neutral_neg, 1]
    preds = np.argmax(pp, axis=1)
    return preds, pp