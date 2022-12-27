# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:12:37 2022

@author: muskr
"""
import cv2
import numpy as np
import os


TP = 0
TN = 0
FP = 0
FN = 0

pred_file = r"./paper_images/output/"
truth_file = r"./paper_images/truth/"
    

for file_name in os.listdir(pred_file):
    
    pred_labels = cv2.imread(pred_file + file_name, 0)
    true_labels = cv2.imread(truth_file + file_name, 0)
    
    
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP += np.sum(np.logical_and(pred_labels == 255, true_labels == 255))
     
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN += np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
     
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP += np.sum(np.logical_and(pred_labels == 255, true_labels == 0))
     
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN += np.sum(np.logical_and(pred_labels == 0, true_labels == 255))
 
Accuracy = (TP + TN)/(TP + TN + FP + FN)
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2*(Precision * Recall)/(Precision + Recall)

print("Accuracy: ", Accuracy)
print("Precision: ", Precision)
print("Recall: ", Recall)
print("F1: ",F1)
