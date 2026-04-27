import numpy as np

import network.neuralNet as NN
def evaluate(net: NN.NeuralNet, dataset, verbose=True):
    
    class_names = [] #to avoid unbound errors
    if dataset == "MNIST":
        import setup_datasets.MNIST as ds
    elif dataset =="FASHION_MNIST":
        import setup_datasets.FASHION_MNIST as ds
        class_names = ds.class_names
    elif dataset =="CIFAR10":
        import setup_datasets.CIFAR10 as ds
        class_names = ds.class_names
    else:
        assert False

    x_test, y_test = ds.x_test, ds.y_test
    
    output = net.inference_feedforward(x_test)  # shape: (num_classes, num_samples)

    pred = np.argmax(output, axis=0)
    n = len(y_test)

    # --- Accuracy ---
    accuracy = np.mean(pred == y_test)

    # # --- Cross-entropy loss ---
    # eps = 1e-12  # avoid log(0)
    # probs = np.clip(output, eps, 1 - eps)
    # loss = -np.mean(np.log(probs[y_test, np.arange(n)]))

    # --- Per-class accuracy ---
    classes = np.unique(y_test)
    per_class_acc = {
        int(c): np.mean(pred[y_test == c] == c)
        for c in classes
    }

    # --- Confusion matrix ---
    num_classes = len(classes)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, p in zip(y_test, pred):
        confusion[true][p] += 1

    # --- Precision, Recall, F1 (macro-averaged) ---
    precision, recall, f1 = [], [], []
    for c in classes:
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    metrics = {
        "accuracy":       accuracy,
        # "loss":           loss,
        "per_class_acc":  per_class_acc,
        "confusion":      confusion,
        "precision":      np.mean(precision),
        "recall":         np.mean(recall),
        "f1":             np.mean(f1),
    }
    
    if dataset != "MNIST":
        # print('ba')
        metrics["names"] = class_names

    if verbose:
        _print_metrics(dataset, metrics)

    return metrics


def _print_metrics(dataset: str, m: dict):
    print("=" * 36)
    print(f"  Accuracy   : {m['accuracy']*100:.2f}%")
    # print(f"  Loss       : {m['loss']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}  (macro)")
    print(f"  Recall     : {m['recall']:.4f}  (macro)")
    print(f"  F1 Score   : {m['f1']:.4f}  (macro)")
    
    print("-" * 36)
    
    print("  Per-class accuracy:")
    for cls, acc in m["per_class_acc"].items():
        
        if "names" not in m:
            print(f"    {cls}: {acc*100:.2f}%")
        else:
            # print('b')
            print(f"    {cls}-{m["names"][cls]}: {acc*100:.2f}%")
        
    print("-" * 36)
    
    print("  Confusion Matrix (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{i:>4}" for i in range(len(m["confusion"])))
    print(header)
    for i, row in enumerate(m["confusion"]):
        row_str = "  ".join(f"{v:>4}" for v in row)
        print(f"    {i} | {row_str}")
        
    print("=" * 36)
    
    
if __name__ == "__main__":
    
    import argparse

    from testing import import_params
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--params_folder", default="FINAL_PARAMS")
    args = parser.parse_args()

    net = import_params.NN_from_params(args.dataset, args.params_folder)

    metrics = evaluate(net, dataset=args.dataset, verbose=True)