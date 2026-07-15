import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math, time, os, datetime, psutil, gc
import random
import kagglehub
import warnings
from utility import get_system_stats, EarlyStopping
from validation import run_classifier_evaluation_pipeline


# label_len = math.ceil(seq_len/2)
# batch_size = 32
# mask_type = "None"
# seq_len = 360


def run_training_pipeline(model, train_loader, num_epochs, device, criterion, optimizer, baseline_stats):
    model.train()
    training_stats = []
    baseline_cpu = baseline_stats["cpu_mem_mb"]
    baseline_gpu = baseline_stats["gpu_mem_mb"]
    
    # Calculate batches dynamically from the loader
    num_batches = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        # Metric trackers
        cpu_batch_deltas, gpu_batch_deltas = [], []
        cpu_peak_from_base, gpu_peak_from_base = [], []
        gpu_util_batches = []

        for i, (x_num, x_cat, x_mark, y) in enumerate(train_loader):
            # 1. Device Transfer
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            x_mark = x_mark.to(device)
            
            stats_before = get_system_stats(device, reset=True)
            
            optimizer.zero_grad()

            # 2. Forward Pass: Pass split inputs
            # The Informer model now expects these separate streams
            recon = model(
                x_num_enc=x_num, x_cat_enc=x_cat, x_mark_enc=x_mark,
                x_num_dec=x_num, x_cat_dec=x_cat, x_mark_dec=x_mark
            )
            
            # Loss is calculated against the numerical ground truth
            loss = criterion(recon, x_num)
            loss.backward()
            optimizer.step()

            # 3. Metrics Tracking
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            stats_after = get_system_stats(device)
            cpu_delta = stats_after["cpu_mem_mb"] - stats_before["cpu_mem_mb"]
            gpu_delta = stats_after["gpu_mem_mb"] - stats_before["gpu_mem_mb"]
            
            cpu_batch_deltas.append(cpu_delta)
            gpu_batch_deltas.append(gpu_delta)
            gpu_peak_from_base.append(stats_after["gpu_peak_mb"] - baseline_gpu)
            gpu_util_batches.append(stats_after["gpu_util_percent"])

        # Epoch Summary
        training_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f} | Time: {training_time:.2f}s")
        training_stats.append({"epoch": epoch + 1, "avg_loss": avg_loss})

    return training_stats


def run_classifier_training_earlystop_pipeline(model, train_loader, val_loader, num_epochs, device, criterion, optimizer, baseline_stats):
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True, path=f'../models/best_model.pt')
    
    stats_history = []
    
    for epoch in range(num_epochs):
        model.train()
        # ... [Your standard training loop logic here] ...
        train_loss = 0.0 # Calculate your actual training loss
        
        # 1. Run Validation
        # Using the function we discussed earlier
        val_loss, val_acc = run_classifier_evaluation_pipeline(model, val_loader, device)
        
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        stats_history.append(epoch_stats)
        
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 2. Check Early Stopping
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # 3. Load the best model weights before returning
    model.load_state_dict(torch.load('../models/best_model.pt'))
    return stats_history
##############################
def run_classifier_training_pipeline(model, train_loader, num_epochs, device, criterion, optimizer, baseline_stats):
    model.train()
    training_stats = []
    
    # Calculate batches dynamically
    num_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for i, (x_num, x_cat, x_mark, y) in enumerate(train_loader):
            # 1. Device Transfer
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            x_mark = x_mark.to(device)
            y = y.to(device).long() # Labels must be long for CrossEntropyLoss
            
            optimizer.zero_grad()

            # 2. Forward Pass: Classification only needs Encoder inputs
            outputs = model(
                x_num_enc=x_num, 
                x_cat_enc=x_cat, 
                x_mark_enc=x_mark
            )
            
            # 3. Loss & Backprop
            # outputs shape: [Batch, Num_Classes]
            # y shape: [Batch]
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # 4. Accuracy Calculation
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == y.data)
            total_samples += y.size(0)
            
            epoch_loss += loss.item()

        # Epoch Summary
        training_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions.double() / total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Time: {training_time:.2f}s")
        training_stats.append({
            "epoch": epoch + 1, 
            "avg_loss": avg_loss, 
            "accuracy": accuracy.item()
        })

    return training_stats



def run_classifier_training_pipeline_dcnn(model, train_loader, num_epochs, device, criterion, optimizer, baseline_stats):
    model.train()
    training_stats = []
    num_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for i, (x_num, x_cat, x_mark, y) in enumerate(train_loader):
            # Device Transfer
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            x_mark = x_mark.to(device)
            y = y.to(device).long()
            
            optimizer.zero_grad()

            # Forward Pass: Passing keywords matching DCNNLSTM.forward
            outputs = model(
                x_num=x_num, 
                x_cat=x_cat, 
                x_mark=x_mark
            )
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Metrics
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == y.data)
            total_samples += y.size(0)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions.double() / total_samples
        training_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Time: {training_time:.2f}s")
        
        training_stats.append({
            "epoch": epoch + 1, 
            "avg_loss": avg_loss, 
            "accuracy": accuracy.item()
        })

    return training_stats

# losses = []
# all_labels = []
# training_stats = []

# model.train()

# # ---- GLOBAL BASELINE BEFORE TRAINING ----
# baseline_stats = get_system_stats(device)
# baseline_cpu = baseline_stats["cpu_mem_mb"]
# baseline_gpu = baseline_stats["gpu_mem_mb"]

# print(f"Baseline CPU Memory: {baseline_cpu:.2f}MB")
# print(f"Baseline GPU Memory: {baseline_gpu:.2f}MB\n")

# for epoch in range(num_epochs):
#     start_time = time.time()
#     epoch_loss = 0.0

#     cpu_batch_deltas = []
#     cpu_peak_from_base = []

#     gpu_batch_deltas = []
#     gpu_peak_from_base = []
#     gpu_util_batches = []

#     for i, (x_num, x_cat, x_mark, y) in enumerate(train_loader):

#         # ---- BEFORE BATCH ----
#         stats_before = get_system_stats(device, reset=True)

#         cpu_before = stats_before["cpu_mem_mb"]
#         gpu_before = stats_before["gpu_mem_mb"]

#         x_num = x_num.to(device) if x_num is not None else None
#         x_cat = x_cat.to(device) if x_cat is not None else None
#         # x = x.to(device)
#         x_mark = x_mark.to(device)

#         optimizer.zero_grad()

#         recon = model(x, x_mark, x, x_mark) #  Base informer model
#         # recon = model(x, x_mark, x, x_mark, enc_self_mask=mask_type) # Model with Encoder Masking
#         loss = criterion(recon, x)
#         loss.backward()
#         optimizer.step()

#         batch_loss = loss.item()
#         epoch_loss += batch_loss
#         losses.append(batch_loss)
#         all_labels.append(y.cpu())

#         # ---- AFTER BATCH ----
#         stats_after = get_system_stats(device)

#         cpu_after = stats_after["cpu_mem_mb"]
#         gpu_after = stats_after["gpu_mem_mb"]
#         gpu_peak = stats_after["gpu_peak_mb"]
#         gpu_util = stats_after["gpu_util_percent"]

#         # ---- CPU METRICS ----
#         cpu_delta = cpu_after - cpu_before
#         cpu_relative_batch = (cpu_delta / cpu_before) * 100 if cpu_before > 0 else 0

#         cpu_peak_base = cpu_after - baseline_cpu
#         cpu_peak_relative = (cpu_peak_base / baseline_cpu) * 100 if baseline_cpu > 0 else 0

#         cpu_batch_deltas.append(cpu_delta)
#         cpu_peak_from_base.append(cpu_peak_base)

#         # ---- GPU METRICS ----
#         gpu_delta = gpu_after - gpu_before
#         gpu_relative_batch = (gpu_delta / gpu_before) * 100 if gpu_before > 0 else 0

#         gpu_peak_base = gpu_peak - baseline_gpu
#         gpu_peak_relative = (gpu_peak_base / baseline_gpu) * 100 if baseline_gpu > 0 else 0

#         gpu_batch_deltas.append(gpu_delta)
#         gpu_peak_from_base.append(gpu_peak_base)
#         gpu_util_batches.append(gpu_util)

#         print(
#             f"Epoch {epoch+1}/{num_epochs}, "
#             f"Batch {i+1}/{num_batches}, "
#             f"Loss: {batch_loss:.6f} | "
#             f"CPU Δ: {cpu_delta:.2f}MB ({cpu_relative_batch:.2f}%) | "
#             f"CPU Peak vs Base: {cpu_peak_base:.2f}MB ({cpu_peak_relative:.2f}%) | "
#             f"GPU Δ: {gpu_delta:.2f}MB ({gpu_relative_batch:.2f}%) | "
#             f"GPU Peak vs Base: {gpu_peak_base:.2f}MB ({gpu_peak_relative:.2f}%)"
#         )

#     # ---- Epoch Summary ----
#     training_time = time.time() - start_time

#     epoch_summary = {
#         "epoch": epoch + 1,
#         "avg_loss": epoch_loss / num_batches,
#         "training_time_sec": training_time,

#         "mean_cpu_batch_delta_mb": sum(cpu_batch_deltas) / len(cpu_batch_deltas),
#         "peak_cpu_batch_delta_mb": max(cpu_batch_deltas),

#         "mean_cpu_peak_from_base_mb": sum(cpu_peak_from_base) / len(cpu_peak_from_base),
#         "peak_cpu_peak_from_base_mb": max(cpu_peak_from_base),

#         "mean_gpu_batch_delta_mb": sum(gpu_batch_deltas) / len(gpu_batch_deltas),
#         "peak_gpu_batch_delta_mb": max(gpu_batch_deltas),

#         "mean_gpu_peak_from_base_mb": sum(gpu_peak_from_base) / len(gpu_peak_from_base),
#         "peak_gpu_peak_from_base_mb": max(gpu_peak_from_base),

#         "mean_gpu_util_percent": sum(gpu_util_batches) / len(gpu_util_batches),
#         "peak_gpu_util_percent": max(gpu_util_batches)
#     }

#     training_stats.append(epoch_summary)

#     print("\n----- Epoch Summary -----")
#     print(f"Epoch {epoch+1} Average Loss: {epoch_summary['avg_loss']:.6f}")
#     # print(f"Training Time: {training_time:.2f}s")

#     # print(f"CPU Batch Δ → Mean: {epoch_summary['mean_cpu_batch_delta_mb']:.2f}MB | "
#     #       f"Peak: {epoch_summary['peak_cpu_batch_delta_mb']:.2f}MB")

#     # print(f"CPU Peak vs Baseline → Mean: {epoch_summary['mean_cpu_peak_from_base_mb']:.2f}MB | "
#     #       f"Peak: {epoch_summary['peak_cpu_peak_from_base_mb']:.2f}MB")

#     # if torch.cuda.is_available():
#     #     print(f"GPU Batch Δ → Mean: {epoch_summary['mean_gpu_batch_delta_mb']:.2f}MB | "
#     #           f"Peak: {epoch_summary['peak_gpu_batch_delta_mb']:.2f}MB")

#     #     print(f"GPU Peak vs Baseline → Mean: {epoch_summary['mean_gpu_peak_from_base_mb']:.2f}MB | "
#     #           f"Peak: {epoch_summary['peak_gpu_peak_from_base_mb']:.2f}MB")

#     #     print(f"GPU Utilization → Mean: {epoch_summary['mean_gpu_util_percent']:.2f}% | "
#     #           f"Peak: {epoch_summary['peak_gpu_util_percent']:.2f}%")

#     # print("-------------------------\n")


import torch
import torch.nn as nn

class DCNNLSTMModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, num_classes):
        super(DCNNLSTMModel, self).__init__()
        
        # 1. DCNN Layers
        self.cnn = nn.Sequential(
            # Input: [Batch, Features, Seq_Len] -> CNN expects features as 'channels'
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 2. LSTM Layer
        # After two MaxPool1d(2), the sequence length is reduced by 4x
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # 3. Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        # CNN wants: [Batch, Features, Seq_Len]
        x = x.transpose(1, 2)
        
        # Pass through CNN
        x = self.cnn(x)
        
        # Prep for LSTM: [Batch, Reduced_Seq_Len, 128]
        x = x.transpose(1, 2)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the last hidden state for classification
        out = self.fc(lstm_out[:, -1, :])
        return out