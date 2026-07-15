import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F

def run_evaluation_pipeline(model, val_loader, device, quantile=0.95):
    """
    Evaluates the model to calculate the anomaly detection threshold.
    Returns: threshold (float), val_errors (tensor), and val_labels (tensor)
    """
    model.eval()
    val_errors = []
    val_labels = []

    # Using no_grad to save memory and computation
    with torch.no_grad():
        for x_num, x_cat, x_mark, y in val_loader:
            # 1. Device Transfer
            # Check for empty tensors from our SequenceDataset update
            x_num = x_num.to(device) if x_num.shape[-1] > 0 else None
            x_cat = x_cat.to(device) if x_cat.shape[-1] > 0 else None
            x_mark = x_mark.to(device)

            # 2. Forward Pass (Informer reconstruction)
            # We pass the same data as encoder and decoder input for reconstruction
            recon = model(
                x_num_enc=x_num, x_cat_enc=x_cat, x_mark_enc=x_mark,
                x_num_dec=x_num, x_cat_dec=x_cat, x_mark_dec=x_mark
            )

            # 3. Error Calculation (MSE per sample in the batch)
            # We compare the reconstruction only against the numerical input
            # recon shape: [Batch, Seq, Num_Features]
            # x_num shape: [Batch, Seq, Num_Features]
            if x_num is not None:
                # Mean error across sequence length and feature dimension
                # Resulting 'error' shape: [Batch]
                error = torch.mean((recon - x_num) ** 2, dim=(1, 2))
                val_errors.append(error.cpu())
                val_labels.append(y.cpu())

    # 4. Concatenate all batch errors into one long tensor
    val_errors = torch.cat(val_errors)
    val_labels = torch.cat(val_labels)

    # 5. Calculate Threshold
    # This defines the "boundary of normalcy"
    threshold = torch.quantile(val_errors, quantile).item()

    print(f"Evaluation Complete. Threshold ({int(quantile*100)}%): {threshold:.6f}")
    
    return threshold, val_errors, val_labels




####################################
# def run_classifier_evaluation_pipeline(model, val_loader, device):
#     """
#     Evaluates the Classifier on the validation set to monitor performance.
#     Returns: avg_val_loss (float), val_accuracy (float)
#     """
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
    
#     # We use CrossEntropyLoss for validation if that's what we used for training
#     criterion = torch.nn.CrossEntropyLoss()

#     with torch.no_grad():
#         for x_num, x_cat, x_mark, y in val_loader:
#             # 1. Device Transfer
#             x_num = x_num.to(device) if x_num.shape[-1] > 0 else None
#             x_cat = x_cat.to(device) if x_cat.shape[-1] > 0 else None
#             x_mark = x_mark.to(device)
#             y = y.to(device).long()

#             # 2. Forward Pass (Classification)
#             outputs = model(
#                 x_num_enc=x_num, 
#                 x_cat_enc=x_cat, 
#                 x_mark_enc=x_mark
#             )

#             # 3. Calculate Loss and Accuracy
#             loss = criterion(outputs, y)
#             val_loss += loss.item()
            
#             _, predicted = torch.max(outputs, 1)
#             total += y.size(0)
#             correct += (predicted == y).sum().item()

#     avg_val_loss = val_loss / len(val_loader)
#     val_accuracy = correct / total

#     print(f"Validation Complete. Avg Loss: {avg_val_loss:.6f} | Accuracy: {val_accuracy:.4f}")
    
#     return avg_val_loss, val_accuracy



def run_classifier_evaluation_pipeline(model, val_loader, device):
    """
    Enhanced Evaluation: Returns loss, accuracy, and weighted F1-score.
    Also returns raw probabilities for threshold tuning.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] # Storing probabilities for ROC/PR thresholding
    
    # Use the same criterion as training
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_num, x_cat, x_mark, y in val_loader:
            # 1. Device Transfer & Handling empty features
            x_num = x_num.to(device) if x_num is not None and x_num.shape[-1] > 0 else None
            x_cat = x_cat.to(device) if x_cat is not None and x_cat.shape[-1] > 0 else None
            x_mark = x_mark.to(device)
            y = y.to(device).long()

            # 2. Forward Pass
            outputs = model(
                x_num_enc=x_num, 
                x_cat_enc=x_cat, 
                x_mark_enc=x_mark
            )

            # 3. Loss
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            # 4. Probabilities & Predictions
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store for batch-independent metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate Global Metrics
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Weighted F1 is better for IoT datasets where 'Normal' outnumbers 'Attack'
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_recall = recall_score(all_labels, all_preds, average='weighted')
    
    print(f"Validation Complete | Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | F1: {val_f1:.4f}")
    
    # We return the scores and labels so you can find the "Best Threshold" outside
    return avg_val_loss, val_accuracy, val_f1, (np.array(all_probs), np.array(all_labels))




def run_classifier_evaluation_pipeline_dcnn(model, val_loader, device):
    """
    DCNN-LSTM Evaluation: Returns loss, accuracy, and weighted F1-score.
    Uses the correct forward pass arguments for DCNN-LSTM.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] 
    
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_num, x_cat, x_mark, y in val_loader:
            # 1. Device Transfer
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            y = y.to(device).long()

            # 2. Forward Pass (Updated keywords for DCNN-LSTM)
            outputs = model(
                x_num=x_num, 
                x_cat=x_cat, 
                x_mark=x_mark.to(device)
            )

            # 3. Loss
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            # 4. Probabilities & Predictions
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    
    val_accuracy = (all_preds_np == all_labels_np).mean()
    val_f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    
    print(f"Validation Complete | Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | F1: {val_f1:.4f}")
    
    return avg_val_loss, val_accuracy, val_f1, (np.array(all_probs), all_labels_np)