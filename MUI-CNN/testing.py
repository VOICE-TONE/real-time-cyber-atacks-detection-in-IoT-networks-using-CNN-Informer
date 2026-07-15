import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, balanced_accuracy_score
)
import torch
import torch.nn.functional as F

def run_testing_pipeline(model, test_loader, device, threshold):
    """
    Evaluates the model on the test set and uses the threshold to detect anomalies.
    Returns: test_stats (dict), y_pred (tensor), y_true (tensor)
    """
    model.eval()
    test_errors = []
    test_labels = []

    with torch.no_grad():
        for x_num, x_cat, x_mark, y in test_loader:
            # 1. Handle potential empty tensors (same as training/eval)
            x_num = x_num.to(device) if x_num.shape[-1] > 0 else None
            x_cat = x_cat.to(device) if x_cat.shape[-1] > 0 else None
            x_mark = x_mark.to(device)

            # 2. Reconstruction Pass
            recon = model(
                x_num_enc=x_num, x_cat_enc=x_cat, x_mark_enc=x_mark,
                x_num_dec=x_num, x_cat_dec=x_cat, x_mark_dec=x_mark
            )

            # 3. Calculate Error (MSE)
            if x_num is not None:
                # Mean error per sample in batch
                error = torch.mean((recon - x_num) ** 2, dim=(1, 2))
                test_errors.append(error.cpu())
                test_labels.append(y.cpu())

    # 4. Concatenate results
    test_errors = torch.cat(test_errors)
    y_true = torch.cat(test_labels)

    # 5. Apply Threshold to get Predictions
    # If error > threshold, it's an anomaly (1), otherwise normal (0)
    y_pred = (test_errors > threshold).int()

    # 6. Generate Performance Metrics
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"Test F1-Score: {f1:.4f}")
    
    test_stats = {
        "f1_score": f1,
        "accuracy": report['1']['accuracy'] if '1' in report else 0,
        "precision": report['1']['precision'] if '1' in report else 0,
        "recall": report['1']['recall'] if '1' in report else 0,
        "errors": test_errors
    }

    return test_stats, y_pred, y_true, test_errors





# def run_classifier_testing_pipeline(model, test_loader, device):
#     """
#     Evaluates the InformerClassifier on the test set.
#     """
#     model.eval()
#     all_preds = []
#     all_true = []

#     with torch.no_grad():
#         for x_num, x_cat, x_mark, y in test_loader:
#             # 1. Device Transfer
#             x_num = x_num.to(device) if x_num.shape[-1] > 0 else None
#             x_cat = x_cat.to(device) if x_cat.shape[-1] > 0 else None
#             x_mark = x_mark.to(device)
#             y = y.to(device)

#             # 2. Forward Pass (Classification)
#             # Output shape: [Batch, Num_Classes]
#             outputs = model(
#                 x_num_enc=x_num, 
#                 x_cat_enc=x_cat, 
#                 x_mark_enc=x_mark
#             )

#             # 3. Get Predictions
#             # torch.max returns (values, indices). Indices are the class IDs.
#             _, preds = torch.max(outputs, 1)

#             all_preds.append(preds.cpu())
#             all_true.append(y.cpu())

#     # 4. Concatenate results
#     y_pred = torch.cat(all_preds).numpy()
#     y_true = torch.cat(all_true).numpy()

#     # 5. Generate Performance Metrics
#     # Using 'macro' or 'weighted' for multi-class DDoS detection
#     f1 = f1_score(y_true, y_pred, average='weighted') 
#     acc = accuracy_score(y_true, y_pred)
    
#     # Precision/Recall per class (Normal, Syn Flood, etc.)
#     report = classification_report(y_true, y_pred, output_dict=True)
#     conf_matrix = confusion_matrix(y_true, y_pred)
    
#     print(f"Test Accuracy: {acc:.4f}")
#     print(f"Test F1-Score (Weighted): {f1:.4f}")
    
#     test_stats = {
#         "f1_score": f1,
#         "accuracy": acc,
#         "confusion_matrix": conf_matrix,
#         "report": report
#     }

#     return test_stats, y_pred, y_true


# def run_classifier_testing_pipeline(model, test_loader, device, threshold=0.5):
#     model.eval()
#     all_preds = []
#     all_true = []
#     all_probs = []

#     with torch.no_grad():
#         for x_num, x_cat, x_mark, y in test_loader:
#             x_num = x_num.to(device) if x_num.shape[-1] > 0 else None
#             x_cat = x_cat.to(device) if x_cat.shape[-1] > 0 else None
#             x_mark = x_mark.to(device)
#             y = y.to(device)

#             outputs = model(
#                 x_num_enc=x_num, 
#                 x_cat_enc=x_cat, 
#                 x_mark_enc=x_mark
#             )

#             # --- NEW THRESHOLD LOGIC ---
#             # 1. Convert logits to probabilities
#             probs = torch.softmax(outputs, dim=1) 
            
#             # 2. Get the probability for the 'Attack' class (index 1)
#             # If your dataset has multiple attack types, this logic changes slightly
#             attack_probs = probs[:, 1] 
            
#             # 3. Apply your custom validation threshold
#             preds = (attack_probs >= threshold).long()
#             # ---------------------------

#             all_preds.append(preds.cpu())
#             all_true.append(y.cpu())
#             all_probs.append(attack_probs.cpu())

#     # ... [Rest of the function remains the same] ...
#     y_pred = torch.cat(all_preds).numpy()
#     y_true = torch.cat(all_true).numpy()
#     y_prob = torch.cat(all_probs).numpy()

#     y_true = y_true.ravel()
#     y_pred = y_pred.ravel()

#     # 5. Generate Performance Metrics    
#     f1 = f1_score(y_true, y_pred, average='weighted')
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, average='weighted')
#     rec = recall_score(y_true, y_pred, average='weighted')
    
#     report = classification_report(y_true, y_pred, output_dict=True)
#     conf_matrix = confusion_matrix(y_true, y_pred)
    
#     # --- Updated test_stats dictionary ---
#     test_stats = {
#         "f1_score": f1,
#         "accuracy": acc,
#         "precision": prec,
#         "recall": rec,
#         "confusion_matrix": conf_matrix,
#         "report": report
#     }

#     print(f"Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

#     return test_stats, y_pred, y_true, y_prob
    
    # Calculate metrics...
    # return test_stats, y_pred, y_true


def run_classifier_testing_pipeline(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []
    test_errors = []

    with torch.no_grad():
        for x_num, x_cat, x_mark, y in test_loader:
            x_num = x_num.to(device) if x_num.shape[-1] > 0 else None
            x_cat = x_cat.to(device) if x_cat.shape[-1] > 0 else None
            x_mark = x_mark.to(device)
            y = y.to(device)

            outputs = model(
                x_num_enc=x_num, 
                x_cat_enc=x_cat, 
                x_mark_enc=x_mark
            )

            # --- THRESHOLD LOGIC ---
            # Using softmax for a 2-column output (Normal vs Attack)
            probs = torch.softmax(outputs, dim=1) 
            attack_probs = probs[:, 1] 
            
            preds = (attack_probs >= threshold).long()
            
            loss_per_sample = torch.nn.functional.cross_entropy(outputs, y, reduction='none')
            test_errors.append(loss_per_sample.cpu())

            all_preds.append(preds.cpu())
            all_true.append(y.cpu())
            all_probs.append(attack_probs.cpu())

    # Concatenate and flatten to 1D arrays
    y_pred = torch.cat(all_preds).numpy().ravel()
    y_true = torch.cat(all_true).numpy().ravel()
    y_prob = torch.cat(all_probs).numpy().ravel()

    # --- GENERATE PERFORMANCE METRICS ---
    # We use zero_division=0 to handle cases where a class might not be predicted
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Extract Specific 'Attack' class metrics (assuming Label 1 is Attack)
    # Use .get() with strings as keys
    attack_key = '1' if '1' in report else '1.0'
    attack_stats = report.get(attack_key, {'precision': 0, 'recall': 0, 'f1-score': 0})

    # Balanced accuracy is the arithmetic mean of class-specific recall
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Macro F1 treats classes equally regardless of sample size
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # --- Updated test_stats dictionary ---
    test_stats = {
        "accuracy": report['accuracy'],
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": report['weighted avg']['f1-score'],
        "precision_attack": attack_stats['precision'],
        "recall_attack": attack_stats['recall'],
        "f1_attack": attack_stats['f1-score'],
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": report
    }

    print("-" * 30)
    print(f"OVERALL ACCURACY:  {test_stats['accuracy']:.4f}")
    print(f"BALANCED ACCURACY: {test_stats['balanced_accuracy']:.4f} (True performance)")
    print(f"MACRO F1 SCORE:    {test_stats['f1_macro']:.4f}")
    print("-" * 30)
    print(f"ATTACK RECALL:     {test_stats['recall_attack']:.4f} (Detection Rate)")
    print(f"ATTACK PRECISION:  {test_stats['precision_attack']:.4f}")
    print("-" * 30)

    return test_stats, y_pred, y_true, y_prob, test_errors


def run_classifier_testing_pipeline_dcnn(model, test_loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_num, x_cat, x_mark, y in test_loader:
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            
            outputs = model(x_num=x_num, x_cat=x_cat, x_mark=x_mark.to(device))
            probs = F.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Anomaly Detection Logic: 1 - P(Normal)
    # Assuming Class 0 is Normal
    anomaly_scores = 1 - all_probs[:, 0]
    y_pred = (anomaly_scores > threshold).astype(int)
    y_true_binary = (all_labels > 0).astype(int)

    report = classification_report(y_true_binary, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true_binary, y_pred)
    
    stats = {
        "accuracy": acc,
        "report": report,
        "balanced_accuracy": balanced_accuracy_score(y_true_binary, y_pred),
        "f1_macro": f1_score(y_true_binary, y_pred, average='macro')
    }
    
    return stats, y_pred, y_true_binary, all_probs, anomaly_scores