

def eval_randomforest(trainx, trainy, testx, testy):
    test_prob = np.zeros(textx.shape[0])
    # your code here
    return {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}

def eval_gbdt(trainx, trainy, testx, testy):
    test_prob = np.zeros(testx.shape[0])
    # your code here
    return {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}

