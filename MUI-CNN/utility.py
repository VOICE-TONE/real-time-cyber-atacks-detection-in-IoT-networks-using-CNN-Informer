from libs import *
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report,precision_recall_curve
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import kagglehub
import warnings


import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        patience (int): How many epochs to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement. 
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




def detect_feature_types(df,
                         max_cat_unique=20,
                         max_cat_ratio=0.01,
                         date_parse_threshold=0.8,
                         sample_size=1000):

    feature_types = {
        "numeric": [],
        "categorical": [],
        "binary": [],
        "datetime": [],
        "id_like": []
    }

    n = len(df)

    for col in df.columns:
        series = df[col]
        num_unique = series.nunique(dropna=False)
        unique_ratio = num_unique / n

        if pd.api.types.is_datetime64_any_dtype(series):
            feature_types["datetime"].append(col)
            continue

        if series.dtype == 'object':
            sample = series.dropna().iloc[:sample_size]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors='coerce')

            success_ratio = parsed.notna().mean()

            if success_ratio > date_parse_threshold:
                feature_types["datetime"].append(col)
            else:
                feature_types["categorical"].append(col)
            continue

        if num_unique == 2:
            feature_types["binary"].append(col)
            continue

        if num_unique == n:
            feature_types["id_like"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):

            if series.min() > 1e9 and series.max() > 1e9:
                try:
                    parsed = pd.to_datetime(series.iloc[:sample_size], unit='s', errors='coerce')
                    if parsed.notna().mean() > 0.9:
                        feature_types["datetime"].append(col)
                        continue
                except:
                    pass

            if (pd.api.types.is_integer_dtype(series) and
                (num_unique <= max_cat_unique or unique_ratio <= max_cat_ratio)):
                feature_types["categorical"].append(col)
                continue

            feature_types["numeric"].append(col)
            continue

        feature_types["numeric"].append(col)

    return feature_types





############
# import numpy as np
def to_numpy(x):
    if hasattr(x, "values"):  # pandas
        return x.values
    return np.asarray(x)



############
# class SequenceDataset(torch.utils.data.Dataset):
#     def __init__(self, X, M, y, seq_len):
#         self.X = X
#         self.M = M
#         self.y = y
#         self.seq_len = seq_len
#         self.nseq = len(X) // seq_len

#     def __len__(self):
#         return self.nseq

#     def __getitem__(self, idx):
#         s = idx * self.seq_len
#         e = s + self.seq_len

#         x = torch.tensor(self.X[s:e], dtype=torch.float32)
#         m = torch.tensor(self.M[s:e], dtype=torch.long)

#         # Sequence-level label:
#         # 1 if ANY anomaly inside sequence
#         y_seq = torch.tensor(
#             int(self.y[s:e].max()),
#             dtype=torch.long
#         )

#         return x, m, y_seq



#########################################
#### Sliding windows without overlap
class SequenceDatasetOverlap(torch.utils.data.Dataset):
    def __init__(self, X, M, y, seq_len, num_indices, cat_indices):
        """
        X: Combined numpy array of all features
        num_indices: List of column indices for numerical features
        cat_indices: List of column indices for categorical features
        """
        self.X = X
        self.M = M
        self.y = y
        self.seq_len = seq_len
        self.num_indices = num_indices
        self.cat_indices = cat_indices
        self.nseq = len(X) // seq_len

    def __len__(self):
        return self.nseq

    def __getitem__(self, idx):
        s = idx * self.seq_len
        e = s + self.seq_len

        # FIX: Instead of None, return an empty tensor [Seq, 0] if indices are empty
        # This allows the DataLoader to collate batches correctly.
        if len(self.num_indices) > 0:
            x_num = torch.tensor(self.X[s:e, self.num_indices], dtype=torch.float32)
        else:
            x_num = torch.zeros((self.seq_len, 0), dtype=torch.float32)

        if len(self.cat_indices) > 0:
            x_cat = torch.tensor(self.X[s:e, self.cat_indices], dtype=torch.long)
        else:
            x_cat = torch.zeros((self.seq_len, 0), dtype=torch.long)
        
        m = torch.tensor(self.M[s:e], dtype=torch.long)

        # Sequence-level label: 1 if ANY anomaly inside sequence
        y_seq = torch.tensor(int(self.y[s:e].max()), dtype=torch.long)

        return x_num, x_cat, m, y_seq
## Defining the process for computing computational efficiency
process = psutil.Process(os.getpid())


#########################################
#### Sliding windows with overlap
class SequenceDatasetNoOverlap(torch.utils.data.Dataset):
    def __init__(self, X, M, y, seq_len, num_indices, cat_indices, stride=None):
        """
        X: Combined numpy array
        stride: If None, defaults to seq_len (disjoint). 
                If smaller than seq_len, windows will overlap.
        """
        self.X = X
        self.M = M
        self.y = y
        self.seq_len = seq_len
        self.num_indices = num_indices
        self.cat_indices = cat_indices
        
        # Use stride if provided, otherwise default to non-overlapping
        self.stride = stride if stride is not None else seq_len
        
        # Calculate total possible windows
        self.num_windows = (len(X) - seq_len) // self.stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # Calculate start and end using stride
        s = idx * self.stride
        e = s + self.seq_len

        # 1. Handle Numerical Features
        if len(self.num_indices) > 0:
            x_num = torch.tensor(self.X[s:e, self.num_indices], dtype=torch.float32)
        else:
            x_num = torch.zeros((self.seq_len, 0), dtype=torch.float32)

        # 2. Handle Categorical Features
        if len(self.cat_indices) > 0:
            x_cat = torch.tensor(self.X[s:e, self.cat_indices], dtype=torch.long)
        else:
            x_cat = torch.zeros((self.seq_len, 0), dtype=torch.long)
        
        # 3. Time Marks (Temporal Embedding)
        m = torch.tensor(self.M[s:e], dtype=torch.long)

        # 4. Label: 1 if ANY anomaly inside sequence (Window Classification)
        y_seq = torch.tensor(int(self.y[s:e].max()), dtype=torch.long)

        return x_num, x_cat, m, y_seq

def get_system_stats(device, reset=False):
    stats = {}
    
    # ---- CPU Memory ----
    stats["cpu_mem_mb"] = process.memory_info().rss / (1024**2)

    # ---- CPU Util ----
    stats["cpu_util_percent"] = psutil.cpu_percent(interval=None)

    # ---- GPU Section ----
    if torch.cuda.is_available() and device.type == "cuda":

        if reset:
            torch.cuda.reset_peak_memory_stats(device)

        stats["gpu_mem_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
        stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            stats["gpu_util_percent"] = util.gpu
        except:
            stats["gpu_util_percent"] = 0

    else:
        stats["gpu_mem_mb"] = 0
        stats["gpu_peak_mb"] = 0
        stats["gpu_util_percent"] = 0

    return stats

import pandas as pd

import pandas as pd

def get_performance_df(model_name, sensor_name, phase, duration, stats):
    """
    Compiles computational metrics for a specific phase (Training or Testing).
    """
    # Define metrics based on the provided stats
    metrics = [
        f"Duration", 
        f"Peak CPU Memory", 
        f"Avg CPU Utilization", 
        f"Peak GPU Memory", 
        f"Avg GPU Utilization"
    ]
    
    values = [
        round(duration, 4),
        round(stats.get("cpu_mem_mb", 0), 2),
        round(stats.get("cpu_util_percent", 0), 2),
        round(stats.get("gpu_peak_mb", 0), 2),
        round(stats.get("gpu_util_percent", 0), 2)
    ]
    
    units = ["Seconds", "MB", "%", "MB", "%"]

    # Build dictionary with metadata
    perf_report = {
        "Metric": metrics,
        "Value": values,
        "Unit": units,
        "model": [model_name] * len(metrics),
        "sensor": [sensor_name] * len(metrics),
        "phase": [phase] * len(metrics)
    }

    return pd.DataFrame(perf_report)


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def get_classification_df(y_true, y_pred, y_probs, class_names, model_name, sensor_name, phase="Testing"):
    """
    Generates a classification report and appends overall Accuracy and AUROC.
    Note: y_probs is required for AUROC calculation.
    """

    full_labels = list(range(len(class_names)))

    # 1. Generate the standard report dictionary
    report_dict = classification_report(
        y_true, 
        y_pred, 
        labels=full_labels,
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    # 2. Convert to DataFrame
    df_metrics = pd.DataFrame(report_dict).transpose().reset_index()
    df_metrics.rename(columns={'index': 'class_or_metric'}, inplace=True)
    
    # 3. Calculate Overall Accuracy and AUROC
    acc_value = accuracy_score(y_true, y_pred)
    
    try:
        # Determine if binary or multi-class for AUROC
        num_classes = len(class_names)
        if num_classes == 2:
            # For binary, use the probabilities of the positive class (column 1)
            auroc_value = roc_auc_score(y_true, np.array(y_probs)[:, 1])
        else:
            # For multi-class, use 'ovr' (One-vs-Rest) with weighted average
            auroc_value = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"⚠️ Could not calculate AUROC: {e}")
        auroc_value = 0.0

    # 4. Append Accuracy and AUROC as new rows
    # We create a small helper DF to append to the main metrics
    extra_rows = pd.DataFrame([
        {"class_or_metric": "overall_accuracy", "precision": acc_value, "recall": acc_value, "f1-score": acc_value, "support": len(y_true)},
        {"class_or_metric": "overall_auroc", "precision": auroc_value, "recall": auroc_value, "f1-score": auroc_value, "support": len(y_true)}
    ])
    
    df_final = pd.concat([df_metrics, extra_rows], ignore_index=True)

    # 5. Add Metadata
    df_final['model'] = model_name
    df_final['sensor'] = sensor_name
    df_final['phase'] = phase
    
    return df_final


def save_report(df, report_type="classification", output_dir="../output/"):
    """
    Generic function to save either classification or performance reports.
    
    Parameters:
    - df: The DataFrame to save.
    - report_type: "classification" or "performance".
    - output_dir: Directory path.
    """
    # 1. Determine the correct filename
    if "classification" in report_type.lower() or "class_or_metric" in df.columns:
        file_name = "classification_report.xlsx"
    else:
        file_name = "computational_performance.xlsx"
    
    # 2. Extract metadata for the sheet name
    # Fallback to "Unknown" if columns are missing
    model = df['model'].iloc[0] if 'model' in df.columns else "Model"
    sensor = df['sensor'].iloc[0] if 'sensor' in df.columns else "Sensor"
    
    sheet_name = f"{model}_{sensor}".strip("_")[:31]

    # 3. Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    full_path = os.path.join(output_dir, file_name)

    # 4. Save/Append logic
    try:
        if not os.path.exists(full_path):
            df.to_excel(full_path, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(full_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Report successfully saved to: {full_path} | Sheet: {sheet_name}")
        
    except PermissionError:
        print(f"❌ Permission Denied: Please close '{file_name}' and try again.")
    except Exception as e:
        print(f"❌ Failed to save {file_name}: {e}")




def Thresholding(anomaliesScore, t='zscore', threshold=2.5):
    """
    Apply statistical thresholding to anomaly scores.

    Parameters:
    -----------
    anomaliesScore : array-like
        Array of anomaly scores.
    t : str
        Method: 'zscore', 'mad', 'chebyshev', 'standard'
    threshold : float
        Threshold value (or percentile if method='standard').

    Returns:
    --------
    finalscoresStatus : np.ndarray
        Array of "Abnormal"/"Normal"
    finalscores : np.ndarray
        Computed standardized scores
    """

    # Convert to numpy if tensor
    if hasattr(anomaliesScore, "detach"):
        anomaliesScore = anomaliesScore.detach().cpu().numpy()

    anomaliesScore = np.asarray(anomaliesScore)
    eps = 1e-8  # numerical stability

    if t == 'mad':
        median = np.median(anomaliesScore)
        mad = np.median(np.abs(anomaliesScore - median)) + eps
        finalscores = 0.6745 * (anomaliesScore - median) / mad

    elif t == 'zscore':
        mean = np.mean(anomaliesScore)
        std = np.std(anomaliesScore) + eps
        finalscores = (anomaliesScore - mean) / std

    elif t == 'chebyshev':
        mean = np.mean(anomaliesScore)
        std = np.std(anomaliesScore) + eps
        finalscores = np.abs(anomaliesScore - mean) / std

    elif t == 'standard':
        #percentile_threshold = np.percentile(anomaliesScore, threshold)
        finalscores = anomaliesScore
        #threshold = percentile_threshold

    else:
        raise ValueError("Choose from: 'zscore', 'mad', 'chebyshev', 'standard'")

    finalscoresStatus = np.where(finalscores >= threshold, "Abnormal", "Normal")

    return finalscoresStatus, finalscores




def model_perf(orig, pred):
    # Force alignment: 0 for Normal, 1 for Abnormal
    # This prevents scikit-learn from alphabetic sorting errors
    true_values = np.where(orig == "Abnormal", 1, np.where(orig == "Normal", 0, orig))
    predicted_values = np.where(pred == "Abnormal", 1, np.where(pred == "Normal", 0, pred))

    # Explicitly set labels to ensure [TN, FP, FN, TP] order
    conf_matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1])
    TN, FP, FN, TP = conf_matrix.ravel()

    # Correct Statistical Calculations
    type_1_error = FP / (FP + TN)  # Probability of a False Alarm
    type_2_error = FN / (FN + TP)  # Probability of Missing an Attack
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Type 1 Error (FPR): {type_1_error:.4f}")
    print(f"Type 2 Error (FNR): {type_2_error:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    auroc = roc_auc_score(true_values, predicted_values) if len(np.unique(true_values)) > 1 else None
    if auroc: print(f"AUROC: {auroc:.4f}")

    class_report = classification_report(true_values, predicted_values, target_names=["Normal", "Abnormal"])
    print("\nClassification Report:")
    print(class_report)

    return {
        "precision": precision, 
        "accuracy": accuracy, 
        "recall": recall, 
        "f1_score": f1, 
        "auroc": auroc,
        "type_1_error": type_1_error
    }

def model_perf_plot(x, y, c, t):
    """
    Parameters:
    - x=xaxis values
    - y=yaxis values
    - c=color column map
    - t=title
    
    Returns:
    - None: Displays the plot

    Example: model_perf_plot(x=[1, 2, 3], y=[3, 2, 1], c=['red', 'blue', 'green'], t="Performance Plot")
    """
    plt.figure(figsize=(18, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.7, c=c, s=1, marker='o', linewidths=0.5)
    plt.title(t)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



    from sklearn.metrics import precision_recall_curve
import numpy as np

def select_best_threshold(probs, labels):
    # We take the probabilities for the 'Attack' class (usually index 1)
    attack_probs = probs[:, 1]
    
    # Calculate precision and recall for every possible threshold
    precisions, recalls, thresholds = precision_recall_curve(labels, attack_probs)
    
    # Calculate F1 for every threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find the index of the highest F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Best Threshold Found: {best_threshold:.4f}")
    print(f"Max F1-Score at this threshold: {f1_scores[best_idx]:.4f}")
    
    return best_threshold

# Usage:
# _, _, _, (val_probs, val_labels) = run_classifier_evaluation_pipeline(...)
# best_thresh = select_best_threshold(val_probs, val_labels)

# def select_best_threshold_anomaly(scores, labels):
#     """
#     scores: Anomaly scores (Higher = More likely an attack)
#     labels: Binary labels (0 = Normal, 1 = Attack)
#     """
#     # If scores are Softmax probabilities, Anomaly Score = 1 - Probability(Normal)
#     if scores.ndim > 1:
#         # Assuming index 0 is 'Normal'
#         scores = 1 - scores[:, 0] 

#     best_f1 = 0
#     best_threshold = 0
#     thresholds = np.linspace(scores.min(), scores.max(), 100)
    
#     for threshold in thresholds:
#         preds = (scores > threshold).astype(int)
#         f1 = f1_score(labels, preds, zero_division=0)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = threshold
            
#     return best_threshold


def select_best_threshold_anomaly(val_losses, val_labels_binary):
    """
    Finds the optimal loss threshold to separate Normal from Attack.
    
    val_losses: 1D array of loss values (one per sequence).
    val_labels_binary: 1D array of binary labels (0=Normal, 1=Attack).
    """
    best_f1 = 0
    best_threshold = 0
    
    # We test thresholds between the minimum and maximum loss seen in validation
    # For DDoS, the max loss might be significantly higher than the mean.
    thresholds = np.linspace(np.min(val_losses), np.max(val_losses), 500)
    
    for threshold in thresholds:
        # Pred 1 if loss > threshold (Anomaly), else 0 (Normal)
        preds = (val_losses > threshold).astype(int)
        
        # Calculate F1. We use binary because val_labels_binary is now [0, 1]
        f1 = f1_score(val_labels_binary, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"--- Threshold Selection Complete ---")
    print(f"Best Loss Threshold: {best_threshold:.6f}")
    print(f"Validation F1-Score: {best_f1:.4f}")
    
    return best_threshold


import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

def create_session_sequences(df, seq_len=32, feature_cols=None):
    """
    Groups data by session and slices them into fixed-length sequences.
    """
    sequences = []
    labels = []
    
    # Define session by 5-tuple
    session_groups = df.groupby(['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto'])
    
    for _, group in session_groups:
        # Extract only the feature columns and the label
        data = group[feature_cols].values
        label = group['label'].iloc[0] # Assumes session label is consistent
        
        # If session is too short, pad it (Pre-padding)
        if len(data) < seq_len:
            pad_width = seq_len - len(data)
            padded_data = np.pad(data, ((pad_width, 0), (0, 0)), mode='constant')
            sequences.append(padded_data)
            labels.append(label)
        else:
            # If session is long, slide a window (no overlap for clean resampling)
            for i in range(0, len(data) - seq_len + 1, seq_len):
                sequences.append(data[i : i + seq_len])
                labels.append(label)
                
    return np.array(sequences), np.array(labels)


def resample_sequences(X, y, strategy='auto', method='smote'):
    """
    Resamples 3D sequences by flattening and rebuilding.
    """
    n_samples, seq_len, n_features = X.shape
    
    # Step 1: Flatten [Samples, Seq, Feat] -> [Samples, Seq * Feat]
    X_flattened = X.reshape(n_samples, -1)
    
    # Step 2: Apply Resampling
    if method == 'smote':
        sampler = SMOTE(sampling_strategy=strategy, random_state=42)
    else:
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=42)
        
    X_res_flat, y_res = sampler.fit_resample(X_flattened, y)
    
    # Step 3: Reshape back to 3D [New_Samples, Seq, Feat]
    X_resampled = X_res_flat.reshape(-1, seq_len, n_features)
    
    return X_resampled, y_res


import numpy as np

def create_sequences(df, seq_len=32, y_col='label'):
    X, y = [], []
    
    # Select only your numerical/encoded features for the Informer
    # Excluding IDs and IP addresses
    feature_cols = ['src_port', 'dst_port', 'proto_enc', 'duration', 'src_bytes', 'dst_bytes'] # Add yours
    
    for _, session in df.groupby('bidirectional_session_id'):
        data = session[feature_cols].values
        # Take the most frequent label in the session as the ground truth
        label = session[y_col].mode()[0] 
        
        # Handle sessions shorter than seq_len with padding
        if len(data) < seq_len:
            pad = np.zeros((seq_len - len(data), len(feature_cols)))
            data = np.vstack([pad, data])
            X.append(data)
            y.append(label)
        else:
            # Slide a window for long sessions
            for i in range(0, len(data) - seq_len + 1, seq_len // 2):
                X.append(data[i : i + seq_len])
                y.append(label)
                
    return np.array(X), np.array(y)