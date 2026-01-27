Attempting to connect to Redis at localhost:6379
Successfully connected to Redis at localhost:6379
Setting up callbacks...
Dash is running on http://127.0.0.1:8062/

 * Serving Flask app 'nearest_neighbor_classifier'
 * Debug mode: off
TEST CALLBACK TRIGGERED: n_clicks=None
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10

Starting Weighted KNN training process... using training dataset (29 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 29
Class distribution in training data: {0: 5, 1: 24}
K Min: 3
K Max: 10
K Step: 1
σ Min: 0.5
σ Max: 2
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 5, 1: 24}
Calibration set class distribution: {0: 2, 1: 8}
Training Weighted KNN 1/32: k=3, σ=0.50, p=1
  K=3, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1050
Training Weighted KNN 2/32: k=3, σ=1.00, p=1
  K=3, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1045
Training Weighted KNN 3/32: k=3, σ=1.50, p=1
  K=3, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1037
Training Weighted KNN 4/32: k=3, σ=2.00, p=1
  K=3, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1026
Training Weighted KNN 5/32: k=4, σ=0.50, p=1
  K=4, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1045
Training Weighted KNN 6/32: k=4, σ=1.00, p=1
  K=4, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1041
Training Weighted KNN 7/32: k=4, σ=1.50, p=1
  K=4, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1036
Training Weighted KNN 8/32: k=4, σ=2.00, p=1
  K=4, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1029
Training Weighted KNN 9/32: k=5, σ=0.50, p=1
  K=5, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1142
Training Weighted KNN 10/32: k=5, σ=1.00, p=1
  K=5, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1141
Training Weighted KNN 11/32: k=5, σ=1.50, p=1
  K=5, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1139
Training Weighted KNN 12/32: k=5, σ=2.00, p=1
  K=5, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1138
Training Weighted KNN 13/32: k=6, σ=0.50, p=1
  K=6, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1152
Training Weighted KNN 14/32: k=6, σ=1.00, p=1
  K=6, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1152
Training Weighted KNN 15/32: k=6, σ=1.50, p=1
  K=6, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1151
Training Weighted KNN 16/32: k=6, σ=2.00, p=1
  K=6, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1151
Training Weighted KNN 17/32: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1128
Training Weighted KNN 18/32: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1127
Training Weighted KNN 19/32: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1125
Training Weighted KNN 20/32: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1124
Training Weighted KNN 21/32: k=8, σ=0.50, p=1
  K=8, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1141
Training Weighted KNN 22/32: k=8, σ=1.00, p=1
  K=8, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1141
Training Weighted KNN 23/32: k=8, σ=1.50, p=1
  K=8, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1140
Training Weighted KNN 24/32: k=8, σ=2.00, p=1
  K=8, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1140
Training Weighted KNN 25/32: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1152
Training Weighted KNN 26/32: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1152
Training Weighted KNN 27/32: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1152
Training Weighted KNN 28/32: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1154
Training Weighted KNN 29/32: k=10, σ=0.50, p=1
  K=10, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.1176
Training Weighted KNN 30/32: k=10, σ=1.00, p=1
  K=10, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.1177
Training Weighted KNN 31/32: k=10, σ=1.50, p=1
  K=10, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.1179
Training Weighted KNN 32/32: k=10, σ=2.00, p=1
  K=10, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.1181

=== Training Complete ===
Process started with PID: 2963
Processing training results

Starting Weighted KNN training process... using training dataset (29 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 29
Class distribution in training data: {0: 5, 1: 24}
K Min: 7
K Max: 13
K Step: 2
σ Min: 0.1
σ Max: 4
σ Step: 0.1
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 5, 1: 24}
Calibration set class distribution: {0: 2, 1: 8}
Training Weighted KNN 1/160: k=7, σ=0.10, p=2
  K=7, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 2/160: k=7, σ=0.20, p=2
  K=7, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 3/160: k=7, σ=0.30, p=2
  K=7, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 4/160: k=7, σ=0.40, p=2
  K=7, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 5/160: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 6/160: k=7, σ=0.60, p=2
  K=7, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 7/160: k=7, σ=0.70, p=2
  K=7, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 8/160: k=7, σ=0.80, p=2
  K=7, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 9/160: k=7, σ=0.90, p=2
  K=7, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 10/160: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 11/160: k=7, σ=1.10, p=2
  K=7, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.1234
Training Weighted KNN 12/160: k=7, σ=1.20, p=2
  K=7, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 13/160: k=7, σ=1.30, p=2
  K=7, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 14/160: k=7, σ=1.40, p=2
  K=7, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 15/160: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.1232
Training Weighted KNN 16/160: k=7, σ=1.60, p=2
  K=7, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.1232
Training Weighted KNN 17/160: k=7, σ=1.70, p=2
  K=7, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 18/160: k=7, σ=1.80, p=2
  K=7, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 19/160: k=7, σ=1.90, p=2
  K=7, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 20/160: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 21/160: k=7, σ=2.10, p=2
  K=7, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 22/160: k=7, σ=2.20, p=2
  K=7, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.1226
Training Weighted KNN 23/160: k=7, σ=2.30, p=2
  K=7, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.1225
Training Weighted KNN 24/160: k=7, σ=2.40, p=2
  K=7, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.1223
Training Weighted KNN 25/160: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.1222
Training Weighted KNN 26/160: k=7, σ=2.60, p=2
  K=7, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.1220
Training Weighted KNN 27/160: k=7, σ=2.70, p=2
  K=7, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.1217
Training Weighted KNN 28/160: k=7, σ=2.80, p=2
  K=7, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.1215
Training Weighted KNN 29/160: k=7, σ=2.90, p=2
  K=7, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.1213
Training Weighted KNN 30/160: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.1210
Training Weighted KNN 31/160: k=7, σ=3.10, p=2
  K=7, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.1207
Training Weighted KNN 32/160: k=7, σ=3.20, p=2
  K=7, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.1204
Training Weighted KNN 33/160: k=7, σ=3.30, p=2
  K=7, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.1201
Training Weighted KNN 34/160: k=7, σ=3.40, p=2
  K=7, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.1198
Training Weighted KNN 35/160: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.1195
Training Weighted KNN 36/160: k=7, σ=3.60, p=2
  K=7, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.1191
Training Weighted KNN 37/160: k=7, σ=3.70, p=2
  K=7, σ=3.70, p=2: Calibration Acc=0.9000, Brier=0.1187
Training Weighted KNN 38/160: k=7, σ=3.80, p=2
  K=7, σ=3.80, p=2: Calibration Acc=0.9000, Brier=0.1184
Training Weighted KNN 39/160: k=7, σ=3.90, p=2
  K=7, σ=3.90, p=2: Calibration Acc=0.9000, Brier=0.1180
Training Weighted KNN 40/160: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.1176
Training Weighted KNN 41/160: k=9, σ=0.10, p=2
  K=9, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 42/160: k=9, σ=0.20, p=2
  K=9, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 43/160: k=9, σ=0.30, p=2
  K=9, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 44/160: k=9, σ=0.40, p=2
  K=9, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 45/160: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 46/160: k=9, σ=0.60, p=2
  K=9, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 47/160: k=9, σ=0.70, p=2
  K=9, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 48/160: k=9, σ=0.80, p=2
  K=9, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 49/160: k=9, σ=0.90, p=2
  K=9, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 50/160: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 51/160: k=9, σ=1.10, p=2
  K=9, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 52/160: k=9, σ=1.20, p=2
  K=9, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 53/160: k=9, σ=1.30, p=2
  K=9, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 54/160: k=9, σ=1.40, p=2
  K=9, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 55/160: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 56/160: k=9, σ=1.60, p=2
  K=9, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 57/160: k=9, σ=1.70, p=2
  K=9, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 58/160: k=9, σ=1.80, p=2
  K=9, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.1227
Training Weighted KNN 59/160: k=9, σ=1.90, p=2
  K=9, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.1226
Training Weighted KNN 60/160: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.1225
Training Weighted KNN 61/160: k=9, σ=2.10, p=2
  K=9, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.1224
Training Weighted KNN 62/160: k=9, σ=2.20, p=2
  K=9, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.1223
Training Weighted KNN 63/160: k=9, σ=2.30, p=2
  K=9, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.1221
Training Weighted KNN 64/160: k=9, σ=2.40, p=2
  K=9, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.1220
Training Weighted KNN 65/160: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.1218
Training Weighted KNN 66/160: k=9, σ=2.60, p=2
  K=9, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.1216
Training Weighted KNN 67/160: k=9, σ=2.70, p=2
  K=9, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.1214
Training Weighted KNN 68/160: k=9, σ=2.80, p=2
  K=9, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.1212
Training Weighted KNN 69/160: k=9, σ=2.90, p=2
  K=9, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.1209
Training Weighted KNN 70/160: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.1207
Training Weighted KNN 71/160: k=9, σ=3.10, p=2
  K=9, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.1204
Training Weighted KNN 72/160: k=9, σ=3.20, p=2
  K=9, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.1201
Training Weighted KNN 73/160: k=9, σ=3.30, p=2
  K=9, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.1198
Training Weighted KNN 74/160: k=9, σ=3.40, p=2
  K=9, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.1194
Training Weighted KNN 75/160: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.1191
Training Weighted KNN 76/160: k=9, σ=3.60, p=2
  K=9, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.1187
Training Weighted KNN 77/160: k=9, σ=3.70, p=2
  K=9, σ=3.70, p=2: Calibration Acc=0.9000, Brier=0.1184
Training Weighted KNN 78/160: k=9, σ=3.80, p=2
  K=9, σ=3.80, p=2: Calibration Acc=0.9000, Brier=0.1180
Training Weighted KNN 79/160: k=9, σ=3.90, p=2
  K=9, σ=3.90, p=2: Calibration Acc=0.9000, Brier=0.1176
Training Weighted KNN 80/160: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.1172
Training Weighted KNN 81/160: k=11, σ=0.10, p=2
  K=11, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 82/160: k=11, σ=0.20, p=2
  K=11, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 83/160: k=11, σ=0.30, p=2
  K=11, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 84/160: k=11, σ=0.40, p=2
  K=11, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 85/160: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 86/160: k=11, σ=0.60, p=2
  K=11, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 87/160: k=11, σ=0.70, p=2
  K=11, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 88/160: k=11, σ=0.80, p=2
  K=11, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 89/160: k=11, σ=0.90, p=2
  K=11, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 90/160: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 91/160: k=11, σ=1.10, p=2
  K=11, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 92/160: k=11, σ=1.20, p=2
  K=11, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.1233
Training Weighted KNN 93/160: k=11, σ=1.30, p=2
  K=11, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.1232
Training Weighted KNN 94/160: k=11, σ=1.40, p=2
  K=11, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.1232
Training Weighted KNN 95/160: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.1232
Training Weighted KNN 96/160: k=11, σ=1.60, p=2
  K=11, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.1231
Training Weighted KNN 97/160: k=11, σ=1.70, p=2
  K=11, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 98/160: k=11, σ=1.80, p=2
  K=11, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.1230
Training Weighted KNN 99/160: k=11, σ=1.90, p=2
  K=11, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 100/160: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 101/160: k=11, σ=2.10, p=2
  K=11, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.1227
Training Weighted KNN 102/160: k=11, σ=2.20, p=2
  K=11, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.1226
Training Weighted KNN 103/160: k=11, σ=2.30, p=2
  K=11, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.1224
Training Weighted KNN 104/160: k=11, σ=2.40, p=2
  K=11, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.1223
Training Weighted KNN 105/160: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.1221
Training Weighted KNN 106/160: k=11, σ=2.60, p=2
  K=11, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.1219
Training Weighted KNN 107/160: k=11, σ=2.70, p=2
  K=11, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.1217
Training Weighted KNN 108/160: k=11, σ=2.80, p=2
  K=11, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.1215
Training Weighted KNN 109/160: k=11, σ=2.90, p=2
  K=11, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.1212
Training Weighted KNN 110/160: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.1210
Training Weighted KNN 111/160: k=11, σ=3.10, p=2
  K=11, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.1207
Training Weighted KNN 112/160: k=11, σ=3.20, p=2
  K=11, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.1204
Training Weighted KNN 113/160: k=11, σ=3.30, p=2
  K=11, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.1201
Training Weighted KNN 114/160: k=11, σ=3.40, p=2
  K=11, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.1198
Training Weighted KNN 115/160: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.1195
Training Weighted KNN 116/160: k=11, σ=3.60, p=2
  K=11, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.1191
Training Weighted KNN 117/160: k=11, σ=3.70, p=2
  K=11, σ=3.70, p=2: Calibration Acc=0.9000, Brier=0.1188
Training Weighted KNN 118/160: k=11, σ=3.80, p=2
  K=11, σ=3.80, p=2: Calibration Acc=0.9000, Brier=0.1184
Training Weighted KNN 119/160: k=11, σ=3.90, p=2
  K=11, σ=3.90, p=2: Calibration Acc=0.9000, Brier=0.1180
Training Weighted KNN 120/160: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.1176
Training Weighted KNN 121/160: k=13, σ=0.10, p=2
  K=13, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 122/160: k=13, σ=0.20, p=2
  K=13, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 123/160: k=13, σ=0.30, p=2
  K=13, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 124/160: k=13, σ=0.40, p=2
  K=13, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 125/160: k=13, σ=0.50, p=2
  K=13, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 126/160: k=13, σ=0.60, p=2
  K=13, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 127/160: k=13, σ=0.70, p=2
  K=13, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 128/160: k=13, σ=0.80, p=2
  K=13, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 129/160: k=13, σ=0.90, p=2
  K=13, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 130/160: k=13, σ=1.00, p=2
  K=13, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.1229
Training Weighted KNN 131/160: k=13, σ=1.10, p=2
  K=13, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 132/160: k=13, σ=1.20, p=2
  K=13, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 133/160: k=13, σ=1.30, p=2
  K=13, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 134/160: k=13, σ=1.40, p=2
  K=13, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.1228
Training Weighted KNN 135/160: k=13, σ=1.50, p=2
  K=13, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.1227
Training Weighted KNN 136/160: k=13, σ=1.60, p=2
  K=13, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.1227
Training Weighted KNN 137/160: k=13, σ=1.70, p=2
  K=13, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.1226
Training Weighted KNN 138/160: k=13, σ=1.80, p=2
  K=13, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.1225
Training Weighted KNN 139/160: k=13, σ=1.90, p=2
  K=13, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.1225
Training Weighted KNN 140/160: k=13, σ=2.00, p=2
  K=13, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.1224
Training Weighted KNN 141/160: k=13, σ=2.10, p=2
  K=13, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.1222
Training Weighted KNN 142/160: k=13, σ=2.20, p=2
  K=13, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.1221
Training Weighted KNN 143/160: k=13, σ=2.30, p=2
  K=13, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.1220
Training Weighted KNN 144/160: k=13, σ=2.40, p=2
  K=13, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.1218
Training Weighted KNN 145/160: k=13, σ=2.50, p=2
  K=13, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.1216
Training Weighted KNN 146/160: k=13, σ=2.60, p=2
  K=13, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.1215
Training Weighted KNN 147/160: k=13, σ=2.70, p=2
  K=13, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.1213
Training Weighted KNN 148/160: k=13, σ=2.80, p=2
  K=13, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.1210
Training Weighted KNN 149/160: k=13, σ=2.90, p=2
  K=13, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.1208
Training Weighted KNN 150/160: k=13, σ=3.00, p=2
  K=13, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.1205
Training Weighted KNN 151/160: k=13, σ=3.10, p=2
  K=13, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.1203
Training Weighted KNN 152/160: k=13, σ=3.20, p=2
  K=13, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.1200
Training Weighted KNN 153/160: k=13, σ=3.30, p=2
  K=13, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.1197
Training Weighted KNN 154/160: k=13, σ=3.40, p=2
  K=13, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.1193
Training Weighted KNN 155/160: k=13, σ=3.50, p=2
  K=13, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.1190
Training Weighted KNN 156/160: k=13, σ=3.60, p=2
  K=13, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.1187
Training Weighted KNN 157/160: k=13, σ=3.70, p=2
  K=13, σ=3.70, p=2: Calibration Acc=0.9000, Brier=0.1183
Training Weighted KNN 158/160: k=13, σ=3.80, p=2
  K=13, σ=3.80, p=2: Calibration Acc=0.9000, Brier=0.1179
Training Weighted KNN 159/160: k=13, σ=3.90, p=2
  K=13, σ=3.90, p=2: Calibration Acc=0.9000, Brier=0.1176
Training Weighted KNN 160/160: k=13, σ=4.00, p=2
  K=13, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.1172

=== Training Complete ===
Process started with PID: 2971
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {0: 2, 1: 8}
Minimum samples per class: 2

Class 1 calibration scores (8 points):
  Min: 0.5666
  Max: 1.0000
  Mean: 0.8848
  Std: 0.1831
  Percentiles: 25%=0.8723, 50%=0.9859, 75%=0.9979, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (2 points):
  Min: 0.2594
  Max: 0.8897
  Mean: 0.5746
  Std: 0.3152

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                            ●
                            ●
                                                ●
                                                 ●
                                                 ●
                                                 ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.5665580716095119, 0.5694185192749975, 0.9733160282421428, 0.9856099175101416, 0.9862607186289274, 0.9972398692073251, 1.0, 1.0]
Class 0 scores (sorted): [0.2594069772569623, 0.8897464018521729]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.11025359814782698, 1: 0.5665580716095119}
  Class 1 threshold: 0.5665580716095119
  α=0.01: threshold=0.5666
  α=0.01: storing threshold=0.5666
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.11025359814782698, 1: 0.5665580716095119}
  Class 1 threshold: 0.5665580716095119
  α=0.05: threshold=0.5666
  α=0.05: storing threshold=0.5666
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.11025359814782698, 1: 0.5665580716095119}
  Class 1 threshold: 0.5665580716095119
  α=0.10: threshold=0.5666
  α=0.10: storing threshold=0.5666
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.11025359814782698, 1: 0.5694185192749975}
  Class 1 threshold: 0.5694185192749975
  α=0.20: threshold=0.5694
  α=0.20: storing threshold=0.5694
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.11025359814782698, 1: 0.5694185192749975}
  Class 1 threshold: 0.5694185192749975
  α=0.25: threshold=0.5694
  α=0.25: storing threshold=0.5694
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {0: 2, 1: 8}

Export: Class 1 calibration scores (8 points):
  Min: 0.5666
  Max: 1.0000
  Mean: 0.8848
  Std: 0.1831
  Percentiles: 25%=0.8723, 50%=0.9859, 75%=0.9979, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (2 points):
  Min: 0.2594
  Max: 0.8897
  Mean: 0.5746
  Std: 0.3152

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
                            ●
                            ●
                                                ●
                                                 ●
                                                 ●
                                                 ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.5665580716095119, 0.5694185192749975, 0.9733160282421428, 0.9856099175101416, 0.9862607186289274, 0.9972398692073251, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.2594069772569623, 0.8897464018521729]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.5665580716095119
Export: α=0.01: threshold=0.5666
Export: α=0.01: storing threshold=0.5666
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.5665580716095119
Export: α=0.05: threshold=0.5666
Export: α=0.05: storing threshold=0.5666
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.5665580716095119
Export: α=0.10: threshold=0.5666
Export: α=0.10: storing threshold=0.5666
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.5694185192749975
Export: α=0.20: threshold=0.5694
Export: α=0.20: storing threshold=0.5694
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.5694185192749975
Export: α=0.25: threshold=0.5694
Export: α=0.25: storing threshold=0.5694
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
measured_levels shape: (39,)
measured_class_labels shape: (39,)
measured_indices shape: (39,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 38
measured_indices range: 0 to 264
Level 0: level_mask shape: (39,), sum: 29
Level 1: level_mask shape: (39,), sum: 10
TEST CALLBACK TRIGGERED: n_clicks=1
measured_levels shape: (66,)
measured_class_labels shape: (66,)
measured_indices shape: (66,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 65
measured_indices range: 0 to 277
Level 0: level_mask shape: (66,), sum: 29
Level 1: level_mask shape: (66,), sum: 27
Level 2: level_mask shape: (66,), sum: 10

Starting Weighted KNN training process... using training dataset (56 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 56
Class distribution in training data: {0: 7, 1: 49}
K Min: 7
K Max: 13
K Step: 2
σ Min: 0.1
σ Max: 4
σ Step: 0.1
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 7, 1: 49}
Calibration set class distribution: {1: 10}
Training Weighted KNN 1/160: k=7, σ=0.10, p=2
  K=7, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 2/160: k=7, σ=0.20, p=2
  K=7, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 3/160: k=7, σ=0.30, p=2
  K=7, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 4/160: k=7, σ=0.40, p=2
  K=7, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 5/160: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 6/160: k=7, σ=0.60, p=2
  K=7, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 7/160: k=7, σ=0.70, p=2
  K=7, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 8/160: k=7, σ=0.80, p=2
  K=7, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 9/160: k=7, σ=0.90, p=2
  K=7, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 10/160: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 11/160: k=7, σ=1.10, p=2
  K=7, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 12/160: k=7, σ=1.20, p=2
  K=7, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 13/160: k=7, σ=1.30, p=2
  K=7, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 14/160: k=7, σ=1.40, p=2
  K=7, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 15/160: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 16/160: k=7, σ=1.60, p=2
  K=7, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 17/160: k=7, σ=1.70, p=2
  K=7, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 18/160: k=7, σ=1.80, p=2
  K=7, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 19/160: k=7, σ=1.90, p=2
  K=7, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 20/160: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 21/160: k=7, σ=2.10, p=2
  K=7, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 22/160: k=7, σ=2.20, p=2
  K=7, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 23/160: k=7, σ=2.30, p=2
  K=7, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 24/160: k=7, σ=2.40, p=2
  K=7, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 25/160: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 26/160: k=7, σ=2.60, p=2
  K=7, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 27/160: k=7, σ=2.70, p=2
  K=7, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 28/160: k=7, σ=2.80, p=2
  K=7, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 29/160: k=7, σ=2.90, p=2
  K=7, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 30/160: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 31/160: k=7, σ=3.10, p=2
  K=7, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 32/160: k=7, σ=3.20, p=2
  K=7, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 33/160: k=7, σ=3.30, p=2
  K=7, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 34/160: k=7, σ=3.40, p=2
  K=7, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 35/160: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0028
Training Weighted KNN 36/160: k=7, σ=3.60, p=2
  K=7, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 37/160: k=7, σ=3.70, p=2
  K=7, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 38/160: k=7, σ=3.80, p=2
  K=7, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 39/160: k=7, σ=3.90, p=2
  K=7, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0035
Training Weighted KNN 40/160: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0036
Training Weighted KNN 41/160: k=9, σ=0.10, p=2
  K=9, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 42/160: k=9, σ=0.20, p=2
  K=9, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 43/160: k=9, σ=0.30, p=2
  K=9, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 44/160: k=9, σ=0.40, p=2
  K=9, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 45/160: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 46/160: k=9, σ=0.60, p=2
  K=9, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 47/160: k=9, σ=0.70, p=2
  K=9, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 48/160: k=9, σ=0.80, p=2
  K=9, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 49/160: k=9, σ=0.90, p=2
  K=9, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 50/160: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 51/160: k=9, σ=1.10, p=2
  K=9, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 52/160: k=9, σ=1.20, p=2
  K=9, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 53/160: k=9, σ=1.30, p=2
  K=9, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 54/160: k=9, σ=1.40, p=2
  K=9, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 55/160: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 56/160: k=9, σ=1.60, p=2
  K=9, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 57/160: k=9, σ=1.70, p=2
  K=9, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 58/160: k=9, σ=1.80, p=2
  K=9, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 59/160: k=9, σ=1.90, p=2
  K=9, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 60/160: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 61/160: k=9, σ=2.10, p=2
  K=9, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 62/160: k=9, σ=2.20, p=2
  K=9, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 63/160: k=9, σ=2.30, p=2
  K=9, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 64/160: k=9, σ=2.40, p=2
  K=9, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 65/160: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 66/160: k=9, σ=2.60, p=2
  K=9, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 67/160: k=9, σ=2.70, p=2
  K=9, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 68/160: k=9, σ=2.80, p=2
  K=9, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 69/160: k=9, σ=2.90, p=2
  K=9, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 70/160: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 71/160: k=9, σ=3.10, p=2
  K=9, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 72/160: k=9, σ=3.20, p=2
  K=9, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0023
Training Weighted KNN 73/160: k=9, σ=3.30, p=2
  K=9, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 74/160: k=9, σ=3.40, p=2
  K=9, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 75/160: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 76/160: k=9, σ=3.60, p=2
  K=9, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 77/160: k=9, σ=3.70, p=2
  K=9, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 78/160: k=9, σ=3.80, p=2
  K=9, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 79/160: k=9, σ=3.90, p=2
  K=9, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 80/160: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0035
Training Weighted KNN 81/160: k=11, σ=0.10, p=2
  K=11, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 82/160: k=11, σ=0.20, p=2
  K=11, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 83/160: k=11, σ=0.30, p=2
  K=11, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 84/160: k=11, σ=0.40, p=2
  K=11, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 85/160: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 86/160: k=11, σ=0.60, p=2
  K=11, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 87/160: k=11, σ=0.70, p=2
  K=11, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 88/160: k=11, σ=0.80, p=2
  K=11, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 89/160: k=11, σ=0.90, p=2
  K=11, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 90/160: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 91/160: k=11, σ=1.10, p=2
  K=11, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 92/160: k=11, σ=1.20, p=2
  K=11, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 93/160: k=11, σ=1.30, p=2
  K=11, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 94/160: k=11, σ=1.40, p=2
  K=11, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 95/160: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 96/160: k=11, σ=1.60, p=2
  K=11, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 97/160: k=11, σ=1.70, p=2
  K=11, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 98/160: k=11, σ=1.80, p=2
  K=11, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 99/160: k=11, σ=1.90, p=2
  K=11, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 100/160: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 101/160: k=11, σ=2.10, p=2
  K=11, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 102/160: k=11, σ=2.20, p=2
  K=11, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 103/160: k=11, σ=2.30, p=2
  K=11, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 104/160: k=11, σ=2.40, p=2
  K=11, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 105/160: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 106/160: k=11, σ=2.60, p=2
  K=11, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 107/160: k=11, σ=2.70, p=2
  K=11, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 108/160: k=11, σ=2.80, p=2
  K=11, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 109/160: k=11, σ=2.90, p=2
  K=11, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 110/160: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 111/160: k=11, σ=3.10, p=2
  K=11, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 112/160: k=11, σ=3.20, p=2
  K=11, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 113/160: k=11, σ=3.30, p=2
  K=11, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 114/160: k=11, σ=3.40, p=2
  K=11, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 115/160: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 116/160: k=11, σ=3.60, p=2
  K=11, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0028
Training Weighted KNN 117/160: k=11, σ=3.70, p=2
  K=11, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 118/160: k=11, σ=3.80, p=2
  K=11, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0031
Training Weighted KNN 119/160: k=11, σ=3.90, p=2
  K=11, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 120/160: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 121/160: k=13, σ=0.10, p=2
  K=13, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 122/160: k=13, σ=0.20, p=2
  K=13, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 123/160: k=13, σ=0.30, p=2
  K=13, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 124/160: k=13, σ=0.40, p=2
  K=13, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 125/160: k=13, σ=0.50, p=2
  K=13, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 126/160: k=13, σ=0.60, p=2
  K=13, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 127/160: k=13, σ=0.70, p=2
  K=13, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 128/160: k=13, σ=0.80, p=2
  K=13, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 129/160: k=13, σ=0.90, p=2
  K=13, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 130/160: k=13, σ=1.00, p=2
  K=13, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 131/160: k=13, σ=1.10, p=2
  K=13, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 132/160: k=13, σ=1.20, p=2
  K=13, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 133/160: k=13, σ=1.30, p=2
  K=13, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 134/160: k=13, σ=1.40, p=2
  K=13, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 135/160: k=13, σ=1.50, p=2
  K=13, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 136/160: k=13, σ=1.60, p=2
  K=13, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 137/160: k=13, σ=1.70, p=2
  K=13, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 138/160: k=13, σ=1.80, p=2
  K=13, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 139/160: k=13, σ=1.90, p=2
  K=13, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 140/160: k=13, σ=2.00, p=2
  K=13, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 141/160: k=13, σ=2.10, p=2
  K=13, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 142/160: k=13, σ=2.20, p=2
  K=13, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 143/160: k=13, σ=2.30, p=2
  K=13, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 144/160: k=13, σ=2.40, p=2
  K=13, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 145/160: k=13, σ=2.50, p=2
  K=13, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 146/160: k=13, σ=2.60, p=2
  K=13, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 147/160: k=13, σ=2.70, p=2
  K=13, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 148/160: k=13, σ=2.80, p=2
  K=13, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 149/160: k=13, σ=2.90, p=2
  K=13, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 150/160: k=13, σ=3.00, p=2
  K=13, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 151/160: k=13, σ=3.10, p=2
  K=13, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 152/160: k=13, σ=3.20, p=2
  K=13, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 153/160: k=13, σ=3.30, p=2
  K=13, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0023
Training Weighted KNN 154/160: k=13, σ=3.40, p=2
  K=13, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 155/160: k=13, σ=3.50, p=2
  K=13, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 156/160: k=13, σ=3.60, p=2
  K=13, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 157/160: k=13, σ=3.70, p=2
  K=13, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 158/160: k=13, σ=3.80, p=2
  K=13, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 159/160: k=13, σ=3.90, p=2
  K=13, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0031
Training Weighted KNN 160/160: k=13, σ=4.00, p=2
  K=13, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0032

=== Training Complete ===
Process started with PID: 3109
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.9223
  Max: 1.0000
  Mean: 0.9881
  Std: 0.0234
  Percentiles: 25%=0.9913, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {1: 10}

Export: Class 1 calibration scores (10 points):
  Min: 0.9223
  Max: 1.0000
  Mean: 0.9881
  Std: 0.0234
  Percentiles: 25%=0.9913, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity

Starting Weighted KNN training process... using training dataset (56 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 56
Class distribution in training data: {0: 7, 1: 49}
K Min: 7
K Max: 13
K Step: 2
σ Min: 0.1
σ Max: 4
σ Step: 0.1
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 7, 1: 49}
Calibration set class distribution: {1: 10}
Training Weighted KNN 1/160: k=7, σ=0.10, p=2
  K=7, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 2/160: k=7, σ=0.20, p=2
  K=7, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 3/160: k=7, σ=0.30, p=2
  K=7, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 4/160: k=7, σ=0.40, p=2
  K=7, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 5/160: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 6/160: k=7, σ=0.60, p=2
  K=7, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 7/160: k=7, σ=0.70, p=2
  K=7, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 8/160: k=7, σ=0.80, p=2
  K=7, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 9/160: k=7, σ=0.90, p=2
  K=7, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 10/160: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 11/160: k=7, σ=1.10, p=2
  K=7, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 12/160: k=7, σ=1.20, p=2
  K=7, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 13/160: k=7, σ=1.30, p=2
  K=7, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 14/160: k=7, σ=1.40, p=2
  K=7, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 15/160: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 16/160: k=7, σ=1.60, p=2
  K=7, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 17/160: k=7, σ=1.70, p=2
  K=7, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 18/160: k=7, σ=1.80, p=2
  K=7, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 19/160: k=7, σ=1.90, p=2
  K=7, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 20/160: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 21/160: k=7, σ=2.10, p=2
  K=7, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 22/160: k=7, σ=2.20, p=2
  K=7, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 23/160: k=7, σ=2.30, p=2
  K=7, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 24/160: k=7, σ=2.40, p=2
  K=7, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 25/160: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 26/160: k=7, σ=2.60, p=2
  K=7, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 27/160: k=7, σ=2.70, p=2
  K=7, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 28/160: k=7, σ=2.80, p=2
  K=7, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 29/160: k=7, σ=2.90, p=2
  K=7, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 30/160: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 31/160: k=7, σ=3.10, p=2
  K=7, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 32/160: k=7, σ=3.20, p=2
  K=7, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 33/160: k=7, σ=3.30, p=2
  K=7, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 34/160: k=7, σ=3.40, p=2
  K=7, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 35/160: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0028
Training Weighted KNN 36/160: k=7, σ=3.60, p=2
  K=7, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 37/160: k=7, σ=3.70, p=2
  K=7, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 38/160: k=7, σ=3.80, p=2
  K=7, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 39/160: k=7, σ=3.90, p=2
  K=7, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0035
Training Weighted KNN 40/160: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0036
Training Weighted KNN 41/160: k=9, σ=0.10, p=2
  K=9, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 42/160: k=9, σ=0.20, p=2
  K=9, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 43/160: k=9, σ=0.30, p=2
  K=9, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 44/160: k=9, σ=0.40, p=2
  K=9, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 45/160: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 46/160: k=9, σ=0.60, p=2
  K=9, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 47/160: k=9, σ=0.70, p=2
  K=9, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 48/160: k=9, σ=0.80, p=2
  K=9, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 49/160: k=9, σ=0.90, p=2
  K=9, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 50/160: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 51/160: k=9, σ=1.10, p=2
  K=9, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 52/160: k=9, σ=1.20, p=2
  K=9, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 53/160: k=9, σ=1.30, p=2
  K=9, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 54/160: k=9, σ=1.40, p=2
  K=9, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 55/160: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 56/160: k=9, σ=1.60, p=2
  K=9, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 57/160: k=9, σ=1.70, p=2
  K=9, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 58/160: k=9, σ=1.80, p=2
  K=9, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 59/160: k=9, σ=1.90, p=2
  K=9, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 60/160: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 61/160: k=9, σ=2.10, p=2
  K=9, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 62/160: k=9, σ=2.20, p=2
  K=9, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 63/160: k=9, σ=2.30, p=2
  K=9, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 64/160: k=9, σ=2.40, p=2
  K=9, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 65/160: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 66/160: k=9, σ=2.60, p=2
  K=9, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 67/160: k=9, σ=2.70, p=2
  K=9, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 68/160: k=9, σ=2.80, p=2
  K=9, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 69/160: k=9, σ=2.90, p=2
  K=9, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 70/160: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 71/160: k=9, σ=3.10, p=2
  K=9, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 72/160: k=9, σ=3.20, p=2
  K=9, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0023
Training Weighted KNN 73/160: k=9, σ=3.30, p=2
  K=9, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 74/160: k=9, σ=3.40, p=2
  K=9, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 75/160: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 76/160: k=9, σ=3.60, p=2
  K=9, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 77/160: k=9, σ=3.70, p=2
  K=9, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 78/160: k=9, σ=3.80, p=2
  K=9, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 79/160: k=9, σ=3.90, p=2
  K=9, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 80/160: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0035
Training Weighted KNN 81/160: k=11, σ=0.10, p=2
  K=11, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 82/160: k=11, σ=0.20, p=2
  K=11, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 83/160: k=11, σ=0.30, p=2
  K=11, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 84/160: k=11, σ=0.40, p=2
  K=11, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 85/160: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 86/160: k=11, σ=0.60, p=2
  K=11, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 87/160: k=11, σ=0.70, p=2
  K=11, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 88/160: k=11, σ=0.80, p=2
  K=11, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 89/160: k=11, σ=0.90, p=2
  K=11, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 90/160: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 91/160: k=11, σ=1.10, p=2
  K=11, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 92/160: k=11, σ=1.20, p=2
  K=11, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 93/160: k=11, σ=1.30, p=2
  K=11, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 94/160: k=11, σ=1.40, p=2
  K=11, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 95/160: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 96/160: k=11, σ=1.60, p=2
  K=11, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 97/160: k=11, σ=1.70, p=2
  K=11, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 98/160: k=11, σ=1.80, p=2
  K=11, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 99/160: k=11, σ=1.90, p=2
  K=11, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 100/160: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 101/160: k=11, σ=2.10, p=2
  K=11, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 102/160: k=11, σ=2.20, p=2
  K=11, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 103/160: k=11, σ=2.30, p=2
  K=11, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 104/160: k=11, σ=2.40, p=2
  K=11, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 105/160: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 106/160: k=11, σ=2.60, p=2
  K=11, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 107/160: k=11, σ=2.70, p=2
  K=11, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 108/160: k=11, σ=2.80, p=2
  K=11, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 109/160: k=11, σ=2.90, p=2
  K=11, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 110/160: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 111/160: k=11, σ=3.10, p=2
  K=11, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 112/160: k=11, σ=3.20, p=2
  K=11, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 113/160: k=11, σ=3.30, p=2
  K=11, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 114/160: k=11, σ=3.40, p=2
  K=11, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 115/160: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 116/160: k=11, σ=3.60, p=2
  K=11, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0028
Training Weighted KNN 117/160: k=11, σ=3.70, p=2
  K=11, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 118/160: k=11, σ=3.80, p=2
  K=11, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0031
Training Weighted KNN 119/160: k=11, σ=3.90, p=2
  K=11, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 120/160: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 121/160: k=13, σ=0.10, p=2
  K=13, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 122/160: k=13, σ=0.20, p=2
  K=13, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 123/160: k=13, σ=0.30, p=2
  K=13, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 124/160: k=13, σ=0.40, p=2
  K=13, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 125/160: k=13, σ=0.50, p=2
  K=13, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 126/160: k=13, σ=0.60, p=2
  K=13, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 127/160: k=13, σ=0.70, p=2
  K=13, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 128/160: k=13, σ=0.80, p=2
  K=13, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 129/160: k=13, σ=0.90, p=2
  K=13, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 130/160: k=13, σ=1.00, p=2
  K=13, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 131/160: k=13, σ=1.10, p=2
  K=13, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 132/160: k=13, σ=1.20, p=2
  K=13, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 133/160: k=13, σ=1.30, p=2
  K=13, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 134/160: k=13, σ=1.40, p=2
  K=13, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 135/160: k=13, σ=1.50, p=2
  K=13, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 136/160: k=13, σ=1.60, p=2
  K=13, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 137/160: k=13, σ=1.70, p=2
  K=13, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 138/160: k=13, σ=1.80, p=2
  K=13, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 139/160: k=13, σ=1.90, p=2
  K=13, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 140/160: k=13, σ=2.00, p=2
  K=13, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 141/160: k=13, σ=2.10, p=2
  K=13, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 142/160: k=13, σ=2.20, p=2
  K=13, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 143/160: k=13, σ=2.30, p=2
  K=13, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 144/160: k=13, σ=2.40, p=2
  K=13, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 145/160: k=13, σ=2.50, p=2
  K=13, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 146/160: k=13, σ=2.60, p=2
  K=13, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 147/160: k=13, σ=2.70, p=2
  K=13, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 148/160: k=13, σ=2.80, p=2
  K=13, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 149/160: k=13, σ=2.90, p=2
  K=13, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 150/160: k=13, σ=3.00, p=2
  K=13, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 151/160: k=13, σ=3.10, p=2
  K=13, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 152/160: k=13, σ=3.20, p=2
  K=13, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 153/160: k=13, σ=3.30, p=2
  K=13, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0023
Training Weighted KNN 154/160: k=13, σ=3.40, p=2
  K=13, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 155/160: k=13, σ=3.50, p=2
  K=13, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 156/160: k=13, σ=3.60, p=2
  K=13, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 157/160: k=13, σ=3.70, p=2
  K=13, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 158/160: k=13, σ=3.80, p=2
  K=13, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 159/160: k=13, σ=3.90, p=2
  K=13, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0031
Training Weighted KNN 160/160: k=13, σ=4.00, p=2
  K=13, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0032

=== Training Complete ===
Process started with PID: 3111
Processing training results

Starting Weighted KNN training process... using training dataset (56 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 56
Class distribution in training data: {0: 7, 1: 49}
K Min: 7
K Max: 13
K Step: 2
σ Min: 0.1
σ Max: 4
σ Step: 0.1
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 7, 1: 49}
Calibration set class distribution: {1: 10}
Training Weighted KNN 1/160: k=7, σ=0.10, p=2
  K=7, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 2/160: k=7, σ=0.20, p=2
  K=7, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 3/160: k=7, σ=0.30, p=2
  K=7, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 4/160: k=7, σ=0.40, p=2
  K=7, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 5/160: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 6/160: k=7, σ=0.60, p=2
  K=7, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 7/160: k=7, σ=0.70, p=2
  K=7, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 8/160: k=7, σ=0.80, p=2
  K=7, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 9/160: k=7, σ=0.90, p=2
  K=7, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 10/160: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 11/160: k=7, σ=1.10, p=2
  K=7, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 12/160: k=7, σ=1.20, p=2
  K=7, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 13/160: k=7, σ=1.30, p=2
  K=7, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 14/160: k=7, σ=1.40, p=2
  K=7, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 15/160: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 16/160: k=7, σ=1.60, p=2
  K=7, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 17/160: k=7, σ=1.70, p=2
  K=7, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 18/160: k=7, σ=1.80, p=2
  K=7, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 19/160: k=7, σ=1.90, p=2
  K=7, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 20/160: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 21/160: k=7, σ=2.10, p=2
  K=7, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 22/160: k=7, σ=2.20, p=2
  K=7, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 23/160: k=7, σ=2.30, p=2
  K=7, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 24/160: k=7, σ=2.40, p=2
  K=7, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 25/160: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 26/160: k=7, σ=2.60, p=2
  K=7, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 27/160: k=7, σ=2.70, p=2
  K=7, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 28/160: k=7, σ=2.80, p=2
  K=7, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 29/160: k=7, σ=2.90, p=2
  K=7, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 30/160: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 31/160: k=7, σ=3.10, p=2
  K=7, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 32/160: k=7, σ=3.20, p=2
  K=7, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 33/160: k=7, σ=3.30, p=2
  K=7, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 34/160: k=7, σ=3.40, p=2
  K=7, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 35/160: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0028
Training Weighted KNN 36/160: k=7, σ=3.60, p=2
  K=7, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 37/160: k=7, σ=3.70, p=2
  K=7, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 38/160: k=7, σ=3.80, p=2
  K=7, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 39/160: k=7, σ=3.90, p=2
  K=7, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0035
Training Weighted KNN 40/160: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0036
Training Weighted KNN 41/160: k=9, σ=0.10, p=2
  K=9, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 42/160: k=9, σ=0.20, p=2
  K=9, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 43/160: k=9, σ=0.30, p=2
  K=9, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 44/160: k=9, σ=0.40, p=2
  K=9, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 45/160: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 46/160: k=9, σ=0.60, p=2
  K=9, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 47/160: k=9, σ=0.70, p=2
  K=9, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 48/160: k=9, σ=0.80, p=2
  K=9, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 49/160: k=9, σ=0.90, p=2
  K=9, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 50/160: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 51/160: k=9, σ=1.10, p=2
  K=9, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 52/160: k=9, σ=1.20, p=2
  K=9, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 53/160: k=9, σ=1.30, p=2
  K=9, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 54/160: k=9, σ=1.40, p=2
  K=9, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 55/160: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 56/160: k=9, σ=1.60, p=2
  K=9, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 57/160: k=9, σ=1.70, p=2
  K=9, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 58/160: k=9, σ=1.80, p=2
  K=9, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 59/160: k=9, σ=1.90, p=2
  K=9, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 60/160: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 61/160: k=9, σ=2.10, p=2
  K=9, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 62/160: k=9, σ=2.20, p=2
  K=9, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 63/160: k=9, σ=2.30, p=2
  K=9, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 64/160: k=9, σ=2.40, p=2
  K=9, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 65/160: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 66/160: k=9, σ=2.60, p=2
  K=9, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 67/160: k=9, σ=2.70, p=2
  K=9, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 68/160: k=9, σ=2.80, p=2
  K=9, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 69/160: k=9, σ=2.90, p=2
  K=9, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 70/160: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 71/160: k=9, σ=3.10, p=2
  K=9, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 72/160: k=9, σ=3.20, p=2
  K=9, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0023
Training Weighted KNN 73/160: k=9, σ=3.30, p=2
  K=9, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 74/160: k=9, σ=3.40, p=2
  K=9, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 75/160: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 76/160: k=9, σ=3.60, p=2
  K=9, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 77/160: k=9, σ=3.70, p=2
  K=9, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 78/160: k=9, σ=3.80, p=2
  K=9, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 79/160: k=9, σ=3.90, p=2
  K=9, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 80/160: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0035
Training Weighted KNN 81/160: k=11, σ=0.10, p=2
  K=11, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 82/160: k=11, σ=0.20, p=2
  K=11, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 83/160: k=11, σ=0.30, p=2
  K=11, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 84/160: k=11, σ=0.40, p=2
  K=11, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 85/160: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 86/160: k=11, σ=0.60, p=2
  K=11, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 87/160: k=11, σ=0.70, p=2
  K=11, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 88/160: k=11, σ=0.80, p=2
  K=11, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 89/160: k=11, σ=0.90, p=2
  K=11, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 90/160: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 91/160: k=11, σ=1.10, p=2
  K=11, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 92/160: k=11, σ=1.20, p=2
  K=11, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 93/160: k=11, σ=1.30, p=2
  K=11, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 94/160: k=11, σ=1.40, p=2
  K=11, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 95/160: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 96/160: k=11, σ=1.60, p=2
  K=11, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 97/160: k=11, σ=1.70, p=2
  K=11, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 98/160: k=11, σ=1.80, p=2
  K=11, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 99/160: k=11, σ=1.90, p=2
  K=11, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 100/160: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 101/160: k=11, σ=2.10, p=2
  K=11, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 102/160: k=11, σ=2.20, p=2
  K=11, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 103/160: k=11, σ=2.30, p=2
  K=11, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 104/160: k=11, σ=2.40, p=2
  K=11, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 105/160: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 106/160: k=11, σ=2.60, p=2
  K=11, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 107/160: k=11, σ=2.70, p=2
  K=11, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 108/160: k=11, σ=2.80, p=2
  K=11, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 109/160: k=11, σ=2.90, p=2
  K=11, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 110/160: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 111/160: k=11, σ=3.10, p=2
  K=11, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 112/160: k=11, σ=3.20, p=2
  K=11, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 113/160: k=11, σ=3.30, p=2
  K=11, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0024
Training Weighted KNN 114/160: k=11, σ=3.40, p=2
  K=11, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 115/160: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 116/160: k=11, σ=3.60, p=2
  K=11, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0028
Training Weighted KNN 117/160: k=11, σ=3.70, p=2
  K=11, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 118/160: k=11, σ=3.80, p=2
  K=11, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0031
Training Weighted KNN 119/160: k=11, σ=3.90, p=2
  K=11, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0032
Training Weighted KNN 120/160: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0033
Training Weighted KNN 121/160: k=13, σ=0.10, p=2
  K=13, σ=0.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 122/160: k=13, σ=0.20, p=2
  K=13, σ=0.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 123/160: k=13, σ=0.30, p=2
  K=13, σ=0.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 124/160: k=13, σ=0.40, p=2
  K=13, σ=0.40, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 125/160: k=13, σ=0.50, p=2
  K=13, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 126/160: k=13, σ=0.60, p=2
  K=13, σ=0.60, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 127/160: k=13, σ=0.70, p=2
  K=13, σ=0.70, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 128/160: k=13, σ=0.80, p=2
  K=13, σ=0.80, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 129/160: k=13, σ=0.90, p=2
  K=13, σ=0.90, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 130/160: k=13, σ=1.00, p=2
  K=13, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 131/160: k=13, σ=1.10, p=2
  K=13, σ=1.10, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 132/160: k=13, σ=1.20, p=2
  K=13, σ=1.20, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 133/160: k=13, σ=1.30, p=2
  K=13, σ=1.30, p=2: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 134/160: k=13, σ=1.40, p=2
  K=13, σ=1.40, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 135/160: k=13, σ=1.50, p=2
  K=13, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 136/160: k=13, σ=1.60, p=2
  K=13, σ=1.60, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 137/160: k=13, σ=1.70, p=2
  K=13, σ=1.70, p=2: Calibration Acc=1.0000, Brier=0.0008
Training Weighted KNN 138/160: k=13, σ=1.80, p=2
  K=13, σ=1.80, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 139/160: k=13, σ=1.90, p=2
  K=13, σ=1.90, p=2: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 140/160: k=13, σ=2.00, p=2
  K=13, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 141/160: k=13, σ=2.10, p=2
  K=13, σ=2.10, p=2: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 142/160: k=13, σ=2.20, p=2
  K=13, σ=2.20, p=2: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 143/160: k=13, σ=2.30, p=2
  K=13, σ=2.30, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 144/160: k=13, σ=2.40, p=2
  K=13, σ=2.40, p=2: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 145/160: k=13, σ=2.50, p=2
  K=13, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 146/160: k=13, σ=2.60, p=2
  K=13, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 147/160: k=13, σ=2.70, p=2
  K=13, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 148/160: k=13, σ=2.80, p=2
  K=13, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 149/160: k=13, σ=2.90, p=2
  K=13, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 150/160: k=13, σ=3.00, p=2
  K=13, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 151/160: k=13, σ=3.10, p=2
  K=13, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 152/160: k=13, σ=3.20, p=2
  K=13, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0022
Training Weighted KNN 153/160: k=13, σ=3.30, p=2
  K=13, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0023
Training Weighted KNN 154/160: k=13, σ=3.40, p=2
  K=13, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0025
Training Weighted KNN 155/160: k=13, σ=3.50, p=2
  K=13, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0026
Training Weighted KNN 156/160: k=13, σ=3.60, p=2
  K=13, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0027
Training Weighted KNN 157/160: k=13, σ=3.70, p=2
  K=13, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0029
Training Weighted KNN 158/160: k=13, σ=3.80, p=2
  K=13, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0030
Training Weighted KNN 159/160: k=13, σ=3.90, p=2
  K=13, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0031
Training Weighted KNN 160/160: k=13, σ=4.00, p=2
  K=13, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0032

=== Training Complete ===
Process started with PID: 3112
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.9223
  Max: 1.0000
  Mean: 0.9881
  Std: 0.0234
  Percentiles: 25%=0.9913, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.9223
  Max: 1.0000
  Mean: 0.9881
  Std: 0.0234
  Percentiles: 25%=0.9913, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {1: 10}

Export: Class 1 calibration scores (10 points):
  Min: 0.9223
  Max: 1.0000
  Mean: 0.9881
  Std: 0.0234
  Percentiles: 25%=0.9913, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.8895
  Max: 1.0000
  Mean: 0.9820
  Std: 0.0336
  Percentiles: 25%=0.9842, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.8966
  Max: 1.0000
  Mean: 0.9834
  Std: 0.0314
  Percentiles: 25%=0.9860, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {1: 10}

Export: Class 1 calibration scores (10 points):
  Min: 0.8966
  Max: 1.0000
  Mean: 0.9834
  Std: 0.0314
  Percentiles: 25%=0.9860, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity
measured_levels shape: (66,)
measured_class_labels shape: (66,)
measured_indices shape: (66,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 65
measured_indices range: 0 to 277
Level 0: level_mask shape: (66,), sum: 29
Level 1: level_mask shape: (66,), sum: 27
Level 2: level_mask shape: (66,), sum: 10

Starting Weighted KNN training process... using training dataset (56 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 56
Class distribution in training data: {0: 14, 1: 42}
K Min: 7
K Max: 13
K Step: 2
σ Min: 0.1
σ Max: 4
σ Step: 0.1
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 14, 1: 42}
Calibration set class distribution: {1: 10}
Training Weighted KNN 1/160: k=7, σ=0.10, p=2
  K=7, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 2/160: k=7, σ=0.20, p=2
  K=7, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 3/160: k=7, σ=0.30, p=2
  K=7, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 4/160: k=7, σ=0.40, p=2
  K=7, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 5/160: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 6/160: k=7, σ=0.60, p=2
  K=7, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 7/160: k=7, σ=0.70, p=2
  K=7, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 8/160: k=7, σ=0.80, p=2
  K=7, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 9/160: k=7, σ=0.90, p=2
  K=7, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.0388
Training Weighted KNN 10/160: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0389
Training Weighted KNN 11/160: k=7, σ=1.10, p=2
  K=7, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.0389
Training Weighted KNN 12/160: k=7, σ=1.20, p=2
  K=7, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.0389
Training Weighted KNN 13/160: k=7, σ=1.30, p=2
  K=7, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0390
Training Weighted KNN 14/160: k=7, σ=1.40, p=2
  K=7, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0390
Training Weighted KNN 15/160: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0391
Training Weighted KNN 16/160: k=7, σ=1.60, p=2
  K=7, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0392
Training Weighted KNN 17/160: k=7, σ=1.70, p=2
  K=7, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0393
Training Weighted KNN 18/160: k=7, σ=1.80, p=2
  K=7, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0394
Training Weighted KNN 19/160: k=7, σ=1.90, p=2
  K=7, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0396
Training Weighted KNN 20/160: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0397
Training Weighted KNN 21/160: k=7, σ=2.10, p=2
  K=7, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0399
Training Weighted KNN 22/160: k=7, σ=2.20, p=2
  K=7, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0402
Training Weighted KNN 23/160: k=7, σ=2.30, p=2
  K=7, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0404
Training Weighted KNN 24/160: k=7, σ=2.40, p=2
  K=7, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0407
Training Weighted KNN 25/160: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0410
Training Weighted KNN 26/160: k=7, σ=2.60, p=2
  K=7, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.0413
Training Weighted KNN 27/160: k=7, σ=2.70, p=2
  K=7, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.0417
Training Weighted KNN 28/160: k=7, σ=2.80, p=2
  K=7, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.0420
Training Weighted KNN 29/160: k=7, σ=2.90, p=2
  K=7, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.0424
Training Weighted KNN 30/160: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0428
Training Weighted KNN 31/160: k=7, σ=3.10, p=2
  K=7, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.0432
Training Weighted KNN 32/160: k=7, σ=3.20, p=2
  K=7, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0436
Training Weighted KNN 33/160: k=7, σ=3.30, p=2
  K=7, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0440
Training Weighted KNN 34/160: k=7, σ=3.40, p=2
  K=7, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0444
Training Weighted KNN 35/160: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0448
Training Weighted KNN 36/160: k=7, σ=3.60, p=2
  K=7, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0452
Training Weighted KNN 37/160: k=7, σ=3.70, p=2
  K=7, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0456
Training Weighted KNN 38/160: k=7, σ=3.80, p=2
  K=7, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0460
Training Weighted KNN 39/160: k=7, σ=3.90, p=2
  K=7, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0464
Training Weighted KNN 40/160: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0467
Training Weighted KNN 41/160: k=9, σ=0.10, p=2
  K=9, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 42/160: k=9, σ=0.20, p=2
  K=9, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 43/160: k=9, σ=0.30, p=2
  K=9, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 44/160: k=9, σ=0.40, p=2
  K=9, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 45/160: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 46/160: k=9, σ=0.60, p=2
  K=9, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 47/160: k=9, σ=0.70, p=2
  K=9, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 48/160: k=9, σ=0.80, p=2
  K=9, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 49/160: k=9, σ=0.90, p=2
  K=9, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 50/160: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 51/160: k=9, σ=1.10, p=2
  K=9, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 52/160: k=9, σ=1.20, p=2
  K=9, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 53/160: k=9, σ=1.30, p=2
  K=9, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0377
Training Weighted KNN 54/160: k=9, σ=1.40, p=2
  K=9, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0377
Training Weighted KNN 55/160: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0378
Training Weighted KNN 56/160: k=9, σ=1.60, p=2
  K=9, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0379
Training Weighted KNN 57/160: k=9, σ=1.70, p=2
  K=9, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0379
Training Weighted KNN 58/160: k=9, σ=1.80, p=2
  K=9, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0381
Training Weighted KNN 59/160: k=9, σ=1.90, p=2
  K=9, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0382
Training Weighted KNN 60/160: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0383
Training Weighted KNN 61/160: k=9, σ=2.10, p=2
  K=9, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0385
Training Weighted KNN 62/160: k=9, σ=2.20, p=2
  K=9, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0387
Training Weighted KNN 63/160: k=9, σ=2.30, p=2
  K=9, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0389
Training Weighted KNN 64/160: k=9, σ=2.40, p=2
  K=9, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0392
Training Weighted KNN 65/160: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0394
Training Weighted KNN 66/160: k=9, σ=2.60, p=2
  K=9, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.0397
Training Weighted KNN 67/160: k=9, σ=2.70, p=2
  K=9, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.0400
Training Weighted KNN 68/160: k=9, σ=2.80, p=2
  K=9, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0404
Training Weighted KNN 69/160: k=9, σ=2.90, p=2
  K=9, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0407
Training Weighted KNN 70/160: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0410
Training Weighted KNN 71/160: k=9, σ=3.10, p=2
  K=9, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0414
Training Weighted KNN 72/160: k=9, σ=3.20, p=2
  K=9, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0418
Training Weighted KNN 73/160: k=9, σ=3.30, p=2
  K=9, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0421
Training Weighted KNN 74/160: k=9, σ=3.40, p=2
  K=9, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0425
Training Weighted KNN 75/160: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0428
Training Weighted KNN 76/160: k=9, σ=3.60, p=2
  K=9, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0432
Training Weighted KNN 77/160: k=9, σ=3.70, p=2
  K=9, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0436
Training Weighted KNN 78/160: k=9, σ=3.80, p=2
  K=9, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0439
Training Weighted KNN 79/160: k=9, σ=3.90, p=2
  K=9, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0442
Training Weighted KNN 80/160: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0446
Training Weighted KNN 81/160: k=11, σ=0.10, p=2
  K=11, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 82/160: k=11, σ=0.20, p=2
  K=11, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 83/160: k=11, σ=0.30, p=2
  K=11, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 84/160: k=11, σ=0.40, p=2
  K=11, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 85/160: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 86/160: k=11, σ=0.60, p=2
  K=11, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 87/160: k=11, σ=0.70, p=2
  K=11, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 88/160: k=11, σ=0.80, p=2
  K=11, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 89/160: k=11, σ=0.90, p=2
  K=11, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 90/160: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 91/160: k=11, σ=1.10, p=2
  K=11, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 92/160: k=11, σ=1.20, p=2
  K=11, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.0369
Training Weighted KNN 93/160: k=11, σ=1.30, p=2
  K=11, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0370
Training Weighted KNN 94/160: k=11, σ=1.40, p=2
  K=11, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0370
Training Weighted KNN 95/160: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0371
Training Weighted KNN 96/160: k=11, σ=1.60, p=2
  K=11, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0371
Training Weighted KNN 97/160: k=11, σ=1.70, p=2
  K=11, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0372
Training Weighted KNN 98/160: k=11, σ=1.80, p=2
  K=11, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0373
Training Weighted KNN 99/160: k=11, σ=1.90, p=2
  K=11, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0374
Training Weighted KNN 100/160: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0375
Training Weighted KNN 101/160: k=11, σ=2.10, p=2
  K=11, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0377
Training Weighted KNN 102/160: k=11, σ=2.20, p=2
  K=11, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0379
Training Weighted KNN 103/160: k=11, σ=2.30, p=2
  K=11, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0380
Training Weighted KNN 104/160: k=11, σ=2.40, p=2
  K=11, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0383
Training Weighted KNN 105/160: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0385
Training Weighted KNN 106/160: k=11, σ=2.60, p=2
  K=11, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.0387
Training Weighted KNN 107/160: k=11, σ=2.70, p=2
  K=11, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0390
Training Weighted KNN 108/160: k=11, σ=2.80, p=2
  K=11, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0393
Training Weighted KNN 109/160: k=11, σ=2.90, p=2
  K=11, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0395
Training Weighted KNN 110/160: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0398
Training Weighted KNN 111/160: k=11, σ=3.10, p=2
  K=11, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0401
Training Weighted KNN 112/160: k=11, σ=3.20, p=2
  K=11, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0404
Training Weighted KNN 113/160: k=11, σ=3.30, p=2
  K=11, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0407
Training Weighted KNN 114/160: k=11, σ=3.40, p=2
  K=11, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0410
Training Weighted KNN 115/160: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0412
Training Weighted KNN 116/160: k=11, σ=3.60, p=2
  K=11, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0415
Training Weighted KNN 117/160: k=11, σ=3.70, p=2
  K=11, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0417
Training Weighted KNN 118/160: k=11, σ=3.80, p=2
  K=11, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0420
Training Weighted KNN 119/160: k=11, σ=3.90, p=2
  K=11, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0422
Training Weighted KNN 120/160: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0424
Training Weighted KNN 121/160: k=13, σ=0.10, p=2
  K=13, σ=0.10, p=2: Calibration Acc=0.9000, Brier=0.0363
Training Weighted KNN 122/160: k=13, σ=0.20, p=2
  K=13, σ=0.20, p=2: Calibration Acc=0.9000, Brier=0.0363
Training Weighted KNN 123/160: k=13, σ=0.30, p=2
  K=13, σ=0.30, p=2: Calibration Acc=0.9000, Brier=0.0363
Training Weighted KNN 124/160: k=13, σ=0.40, p=2
  K=13, σ=0.40, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 125/160: k=13, σ=0.50, p=2
  K=13, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 126/160: k=13, σ=0.60, p=2
  K=13, σ=0.60, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 127/160: k=13, σ=0.70, p=2
  K=13, σ=0.70, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 128/160: k=13, σ=0.80, p=2
  K=13, σ=0.80, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 129/160: k=13, σ=0.90, p=2
  K=13, σ=0.90, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 130/160: k=13, σ=1.00, p=2
  K=13, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 131/160: k=13, σ=1.10, p=2
  K=13, σ=1.10, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 132/160: k=13, σ=1.20, p=2
  K=13, σ=1.20, p=2: Calibration Acc=0.9000, Brier=0.0364
Training Weighted KNN 133/160: k=13, σ=1.30, p=2
  K=13, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0365
Training Weighted KNN 134/160: k=13, σ=1.40, p=2
  K=13, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0365
Training Weighted KNN 135/160: k=13, σ=1.50, p=2
  K=13, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0365
Training Weighted KNN 136/160: k=13, σ=1.60, p=2
  K=13, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0366
Training Weighted KNN 137/160: k=13, σ=1.70, p=2
  K=13, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0367
Training Weighted KNN 138/160: k=13, σ=1.80, p=2
  K=13, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0367
Training Weighted KNN 139/160: k=13, σ=1.90, p=2
  K=13, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0368
Training Weighted KNN 140/160: k=13, σ=2.00, p=2
  K=13, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0370
Training Weighted KNN 141/160: k=13, σ=2.10, p=2
  K=13, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0371
Training Weighted KNN 142/160: k=13, σ=2.20, p=2
  K=13, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0373
Training Weighted KNN 143/160: k=13, σ=2.30, p=2
  K=13, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0374
Training Weighted KNN 144/160: k=13, σ=2.40, p=2
  K=13, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0376
Training Weighted KNN 145/160: k=13, σ=2.50, p=2
  K=13, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0378
Training Weighted KNN 146/160: k=13, σ=2.60, p=2
  K=13, σ=2.60, p=2: Calibration Acc=1.0000, Brier=0.0381
Training Weighted KNN 147/160: k=13, σ=2.70, p=2
  K=13, σ=2.70, p=2: Calibration Acc=1.0000, Brier=0.0383
Training Weighted KNN 148/160: k=13, σ=2.80, p=2
  K=13, σ=2.80, p=2: Calibration Acc=1.0000, Brier=0.0385
Training Weighted KNN 149/160: k=13, σ=2.90, p=2
  K=13, σ=2.90, p=2: Calibration Acc=1.0000, Brier=0.0388
Training Weighted KNN 150/160: k=13, σ=3.00, p=2
  K=13, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0391
Training Weighted KNN 151/160: k=13, σ=3.10, p=2
  K=13, σ=3.10, p=2: Calibration Acc=1.0000, Brier=0.0393
Training Weighted KNN 152/160: k=13, σ=3.20, p=2
  K=13, σ=3.20, p=2: Calibration Acc=1.0000, Brier=0.0396
Training Weighted KNN 153/160: k=13, σ=3.30, p=2
  K=13, σ=3.30, p=2: Calibration Acc=1.0000, Brier=0.0399
Training Weighted KNN 154/160: k=13, σ=3.40, p=2
  K=13, σ=3.40, p=2: Calibration Acc=1.0000, Brier=0.0401
Training Weighted KNN 155/160: k=13, σ=3.50, p=2
  K=13, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0403
Training Weighted KNN 156/160: k=13, σ=3.60, p=2
  K=13, σ=3.60, p=2: Calibration Acc=1.0000, Brier=0.0406
Training Weighted KNN 157/160: k=13, σ=3.70, p=2
  K=13, σ=3.70, p=2: Calibration Acc=1.0000, Brier=0.0408
Training Weighted KNN 158/160: k=13, σ=3.80, p=2
  K=13, σ=3.80, p=2: Calibration Acc=1.0000, Brier=0.0410
Training Weighted KNN 159/160: k=13, σ=3.90, p=2
  K=13, σ=3.90, p=2: Calibration Acc=1.0000, Brier=0.0412
Training Weighted KNN 160/160: k=13, σ=4.00, p=2
  K=13, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0414

=== Training Complete ===
Process started with PID: 3147
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.4552
  Max: 1.0000
  Mean: 0.8859
  Std: 0.1566
  Percentiles: 25%=0.8611, 50%=0.9139, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {1: 10}

Export: Class 1 calibration scores (10 points):
  Min: 0.4552
  Max: 1.0000
  Mean: 0.8859
  Std: 0.1566
  Percentiles: 25%=0.8611, 50%=0.9139, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity

Starting Weighted KNN training process... using training dataset (56 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 56
Class distribution in training data: {0: 14, 1: 42}
K Min: 7
K Max: 13
K Step: 2
σ Min: 0.1
σ Max: 4
σ Step: 0.1
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 14, 1: 42}
Calibration set class distribution: {1: 10}
Training Weighted KNN 1/160: k=7, σ=0.10, p=1
  K=7, σ=0.10, p=1: Calibration Acc=1.0000, Brier=0.0421
Training Weighted KNN 2/160: k=7, σ=0.20, p=1
  K=7, σ=0.20, p=1: Calibration Acc=1.0000, Brier=0.0421
Training Weighted KNN 3/160: k=7, σ=0.30, p=1
  K=7, σ=0.30, p=1: Calibration Acc=1.0000, Brier=0.0422
Training Weighted KNN 4/160: k=7, σ=0.40, p=1
  K=7, σ=0.40, p=1: Calibration Acc=1.0000, Brier=0.0422
Training Weighted KNN 5/160: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0423
Training Weighted KNN 6/160: k=7, σ=0.60, p=1
  K=7, σ=0.60, p=1: Calibration Acc=1.0000, Brier=0.0424
Training Weighted KNN 7/160: k=7, σ=0.70, p=1
  K=7, σ=0.70, p=1: Calibration Acc=1.0000, Brier=0.0425
Training Weighted KNN 8/160: k=7, σ=0.80, p=1
  K=7, σ=0.80, p=1: Calibration Acc=1.0000, Brier=0.0427
Training Weighted KNN 9/160: k=7, σ=0.90, p=1
  K=7, σ=0.90, p=1: Calibration Acc=1.0000, Brier=0.0428
Training Weighted KNN 10/160: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0430
Training Weighted KNN 11/160: k=7, σ=1.10, p=1
  K=7, σ=1.10, p=1: Calibration Acc=1.0000, Brier=0.0431
Training Weighted KNN 12/160: k=7, σ=1.20, p=1
  K=7, σ=1.20, p=1: Calibration Acc=1.0000, Brier=0.0433
Training Weighted KNN 13/160: k=7, σ=1.30, p=1
  K=7, σ=1.30, p=1: Calibration Acc=1.0000, Brier=0.0435
Training Weighted KNN 14/160: k=7, σ=1.40, p=1
  K=7, σ=1.40, p=1: Calibration Acc=1.0000, Brier=0.0437
Training Weighted KNN 15/160: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0439
Training Weighted KNN 16/160: k=7, σ=1.60, p=1
  K=7, σ=1.60, p=1: Calibration Acc=1.0000, Brier=0.0441
Training Weighted KNN 17/160: k=7, σ=1.70, p=1
  K=7, σ=1.70, p=1: Calibration Acc=1.0000, Brier=0.0443
Training Weighted KNN 18/160: k=7, σ=1.80, p=1
  K=7, σ=1.80, p=1: Calibration Acc=1.0000, Brier=0.0446
Training Weighted KNN 19/160: k=7, σ=1.90, p=1
  K=7, σ=1.90, p=1: Calibration Acc=1.0000, Brier=0.0448
Training Weighted KNN 20/160: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0450
Training Weighted KNN 21/160: k=7, σ=2.10, p=1
  K=7, σ=2.10, p=1: Calibration Acc=1.0000, Brier=0.0452
Training Weighted KNN 22/160: k=7, σ=2.20, p=1
  K=7, σ=2.20, p=1: Calibration Acc=1.0000, Brier=0.0455
Training Weighted KNN 23/160: k=7, σ=2.30, p=1
  K=7, σ=2.30, p=1: Calibration Acc=1.0000, Brier=0.0457
Training Weighted KNN 24/160: k=7, σ=2.40, p=1
  K=7, σ=2.40, p=1: Calibration Acc=1.0000, Brier=0.0459
Training Weighted KNN 25/160: k=7, σ=2.50, p=1
  K=7, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0461
Training Weighted KNN 26/160: k=7, σ=2.60, p=1
  K=7, σ=2.60, p=1: Calibration Acc=1.0000, Brier=0.0463
Training Weighted KNN 27/160: k=7, σ=2.70, p=1
  K=7, σ=2.70, p=1: Calibration Acc=1.0000, Brier=0.0466
Training Weighted KNN 28/160: k=7, σ=2.80, p=1
  K=7, σ=2.80, p=1: Calibration Acc=1.0000, Brier=0.0468
Training Weighted KNN 29/160: k=7, σ=2.90, p=1
  K=7, σ=2.90, p=1: Calibration Acc=1.0000, Brier=0.0470
Training Weighted KNN 30/160: k=7, σ=3.00, p=1
  K=7, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0472
Training Weighted KNN 31/160: k=7, σ=3.10, p=1
  K=7, σ=3.10, p=1: Calibration Acc=1.0000, Brier=0.0474
Training Weighted KNN 32/160: k=7, σ=3.20, p=1
  K=7, σ=3.20, p=1: Calibration Acc=1.0000, Brier=0.0476
Training Weighted KNN 33/160: k=7, σ=3.30, p=1
  K=7, σ=3.30, p=1: Calibration Acc=1.0000, Brier=0.0478
Training Weighted KNN 34/160: k=7, σ=3.40, p=1
  K=7, σ=3.40, p=1: Calibration Acc=1.0000, Brier=0.0480
Training Weighted KNN 35/160: k=7, σ=3.50, p=1
  K=7, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0481
Training Weighted KNN 36/160: k=7, σ=3.60, p=1
  K=7, σ=3.60, p=1: Calibration Acc=1.0000, Brier=0.0483
Training Weighted KNN 37/160: k=7, σ=3.70, p=1
  K=7, σ=3.70, p=1: Calibration Acc=1.0000, Brier=0.0485
Training Weighted KNN 38/160: k=7, σ=3.80, p=1
  K=7, σ=3.80, p=1: Calibration Acc=1.0000, Brier=0.0486
Training Weighted KNN 39/160: k=7, σ=3.90, p=1
  K=7, σ=3.90, p=1: Calibration Acc=1.0000, Brier=0.0488
Training Weighted KNN 40/160: k=7, σ=4.00, p=1
  K=7, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0490
Training Weighted KNN 41/160: k=9, σ=0.10, p=1
  K=9, σ=0.10, p=1: Calibration Acc=1.0000, Brier=0.0404
Training Weighted KNN 42/160: k=9, σ=0.20, p=1
  K=9, σ=0.20, p=1: Calibration Acc=1.0000, Brier=0.0404
Training Weighted KNN 43/160: k=9, σ=0.30, p=1
  K=9, σ=0.30, p=1: Calibration Acc=1.0000, Brier=0.0405
Training Weighted KNN 44/160: k=9, σ=0.40, p=1
  K=9, σ=0.40, p=1: Calibration Acc=1.0000, Brier=0.0405
Training Weighted KNN 45/160: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0406
Training Weighted KNN 46/160: k=9, σ=0.60, p=1
  K=9, σ=0.60, p=1: Calibration Acc=1.0000, Brier=0.0407
Training Weighted KNN 47/160: k=9, σ=0.70, p=1
  K=9, σ=0.70, p=1: Calibration Acc=1.0000, Brier=0.0409
Training Weighted KNN 48/160: k=9, σ=0.80, p=1
  K=9, σ=0.80, p=1: Calibration Acc=1.0000, Brier=0.0410
Training Weighted KNN 49/160: k=9, σ=0.90, p=1
  K=9, σ=0.90, p=1: Calibration Acc=1.0000, Brier=0.0412
Training Weighted KNN 50/160: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0413
Training Weighted KNN 51/160: k=9, σ=1.10, p=1
  K=9, σ=1.10, p=1: Calibration Acc=1.0000, Brier=0.0415
Training Weighted KNN 52/160: k=9, σ=1.20, p=1
  K=9, σ=1.20, p=1: Calibration Acc=1.0000, Brier=0.0417
Training Weighted KNN 53/160: k=9, σ=1.30, p=1
  K=9, σ=1.30, p=1: Calibration Acc=1.0000, Brier=0.0419
Training Weighted KNN 54/160: k=9, σ=1.40, p=1
  K=9, σ=1.40, p=1: Calibration Acc=1.0000, Brier=0.0422
Training Weighted KNN 55/160: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0424
Training Weighted KNN 56/160: k=9, σ=1.60, p=1
  K=9, σ=1.60, p=1: Calibration Acc=1.0000, Brier=0.0426
Training Weighted KNN 57/160: k=9, σ=1.70, p=1
  K=9, σ=1.70, p=1: Calibration Acc=1.0000, Brier=0.0429
Training Weighted KNN 58/160: k=9, σ=1.80, p=1
  K=9, σ=1.80, p=1: Calibration Acc=1.0000, Brier=0.0431
Training Weighted KNN 59/160: k=9, σ=1.90, p=1
  K=9, σ=1.90, p=1: Calibration Acc=1.0000, Brier=0.0434
Training Weighted KNN 60/160: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0436
Training Weighted KNN 61/160: k=9, σ=2.10, p=1
  K=9, σ=2.10, p=1: Calibration Acc=1.0000, Brier=0.0439
Training Weighted KNN 62/160: k=9, σ=2.20, p=1
  K=9, σ=2.20, p=1: Calibration Acc=1.0000, Brier=0.0441
Training Weighted KNN 63/160: k=9, σ=2.30, p=1
  K=9, σ=2.30, p=1: Calibration Acc=1.0000, Brier=0.0444
Training Weighted KNN 64/160: k=9, σ=2.40, p=1
  K=9, σ=2.40, p=1: Calibration Acc=1.0000, Brier=0.0446
Training Weighted KNN 65/160: k=9, σ=2.50, p=1
  K=9, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0449
Training Weighted KNN 66/160: k=9, σ=2.60, p=1
  K=9, σ=2.60, p=1: Calibration Acc=1.0000, Brier=0.0451
Training Weighted KNN 67/160: k=9, σ=2.70, p=1
  K=9, σ=2.70, p=1: Calibration Acc=1.0000, Brier=0.0454
Training Weighted KNN 68/160: k=9, σ=2.80, p=1
  K=9, σ=2.80, p=1: Calibration Acc=1.0000, Brier=0.0456
Training Weighted KNN 69/160: k=9, σ=2.90, p=1
  K=9, σ=2.90, p=1: Calibration Acc=1.0000, Brier=0.0459
Training Weighted KNN 70/160: k=9, σ=3.00, p=1
  K=9, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0461
Training Weighted KNN 71/160: k=9, σ=3.10, p=1
  K=9, σ=3.10, p=1: Calibration Acc=1.0000, Brier=0.0463
Training Weighted KNN 72/160: k=9, σ=3.20, p=1
  K=9, σ=3.20, p=1: Calibration Acc=1.0000, Brier=0.0466
Training Weighted KNN 73/160: k=9, σ=3.30, p=1
  K=9, σ=3.30, p=1: Calibration Acc=1.0000, Brier=0.0468
Training Weighted KNN 74/160: k=9, σ=3.40, p=1
  K=9, σ=3.40, p=1: Calibration Acc=1.0000, Brier=0.0470
Training Weighted KNN 75/160: k=9, σ=3.50, p=1
  K=9, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0472
Training Weighted KNN 76/160: k=9, σ=3.60, p=1
  K=9, σ=3.60, p=1: Calibration Acc=1.0000, Brier=0.0474
Training Weighted KNN 77/160: k=9, σ=3.70, p=1
  K=9, σ=3.70, p=1: Calibration Acc=1.0000, Brier=0.0476
Training Weighted KNN 78/160: k=9, σ=3.80, p=1
  K=9, σ=3.80, p=1: Calibration Acc=1.0000, Brier=0.0478
Training Weighted KNN 79/160: k=9, σ=3.90, p=1
  K=9, σ=3.90, p=1: Calibration Acc=1.0000, Brier=0.0480
Training Weighted KNN 80/160: k=9, σ=4.00, p=1
  K=9, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0482
Training Weighted KNN 81/160: k=11, σ=0.10, p=1
  K=11, σ=0.10, p=1: Calibration Acc=1.0000, Brier=0.0369
Training Weighted KNN 82/160: k=11, σ=0.20, p=1
  K=11, σ=0.20, p=1: Calibration Acc=1.0000, Brier=0.0370
Training Weighted KNN 83/160: k=11, σ=0.30, p=1
  K=11, σ=0.30, p=1: Calibration Acc=1.0000, Brier=0.0370
Training Weighted KNN 84/160: k=11, σ=0.40, p=1
  K=11, σ=0.40, p=1: Calibration Acc=1.0000, Brier=0.0370
Training Weighted KNN 85/160: k=11, σ=0.50, p=1
  K=11, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0371
Training Weighted KNN 86/160: k=11, σ=0.60, p=1
  K=11, σ=0.60, p=1: Calibration Acc=1.0000, Brier=0.0372
Training Weighted KNN 87/160: k=11, σ=0.70, p=1
  K=11, σ=0.70, p=1: Calibration Acc=1.0000, Brier=0.0373
Training Weighted KNN 88/160: k=11, σ=0.80, p=1
  K=11, σ=0.80, p=1: Calibration Acc=1.0000, Brier=0.0374
Training Weighted KNN 89/160: k=11, σ=0.90, p=1
  K=11, σ=0.90, p=1: Calibration Acc=1.0000, Brier=0.0375
Training Weighted KNN 90/160: k=11, σ=1.00, p=1
  K=11, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0376
Training Weighted KNN 91/160: k=11, σ=1.10, p=1
  K=11, σ=1.10, p=1: Calibration Acc=1.0000, Brier=0.0377
Training Weighted KNN 92/160: k=11, σ=1.20, p=1
  K=11, σ=1.20, p=1: Calibration Acc=1.0000, Brier=0.0379
Training Weighted KNN 93/160: k=11, σ=1.30, p=1
  K=11, σ=1.30, p=1: Calibration Acc=1.0000, Brier=0.0380
Training Weighted KNN 94/160: k=11, σ=1.40, p=1
  K=11, σ=1.40, p=1: Calibration Acc=1.0000, Brier=0.0382
Training Weighted KNN 95/160: k=11, σ=1.50, p=1
  K=11, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0383
Training Weighted KNN 96/160: k=11, σ=1.60, p=1
  K=11, σ=1.60, p=1: Calibration Acc=1.0000, Brier=0.0385
Training Weighted KNN 97/160: k=11, σ=1.70, p=1
  K=11, σ=1.70, p=1: Calibration Acc=1.0000, Brier=0.0387
Training Weighted KNN 98/160: k=11, σ=1.80, p=1
  K=11, σ=1.80, p=1: Calibration Acc=1.0000, Brier=0.0388
Training Weighted KNN 99/160: k=11, σ=1.90, p=1
  K=11, σ=1.90, p=1: Calibration Acc=1.0000, Brier=0.0390
Training Weighted KNN 100/160: k=11, σ=2.00, p=1
  K=11, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0391
Training Weighted KNN 101/160: k=11, σ=2.10, p=1
  K=11, σ=2.10, p=1: Calibration Acc=1.0000, Brier=0.0393
Training Weighted KNN 102/160: k=11, σ=2.20, p=1
  K=11, σ=2.20, p=1: Calibration Acc=1.0000, Brier=0.0395
Training Weighted KNN 103/160: k=11, σ=2.30, p=1
  K=11, σ=2.30, p=1: Calibration Acc=1.0000, Brier=0.0396
Training Weighted KNN 104/160: k=11, σ=2.40, p=1
  K=11, σ=2.40, p=1: Calibration Acc=1.0000, Brier=0.0398
Training Weighted KNN 105/160: k=11, σ=2.50, p=1
  K=11, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0399
Training Weighted KNN 106/160: k=11, σ=2.60, p=1
  K=11, σ=2.60, p=1: Calibration Acc=1.0000, Brier=0.0401
Training Weighted KNN 107/160: k=11, σ=2.70, p=1
  K=11, σ=2.70, p=1: Calibration Acc=1.0000, Brier=0.0402
Training Weighted KNN 108/160: k=11, σ=2.80, p=1
  K=11, σ=2.80, p=1: Calibration Acc=1.0000, Brier=0.0403
Training Weighted KNN 109/160: k=11, σ=2.90, p=1
  K=11, σ=2.90, p=1: Calibration Acc=1.0000, Brier=0.0405
Training Weighted KNN 110/160: k=11, σ=3.00, p=1
  K=11, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0406
Training Weighted KNN 111/160: k=11, σ=3.10, p=1
  K=11, σ=3.10, p=1: Calibration Acc=1.0000, Brier=0.0407
Training Weighted KNN 112/160: k=11, σ=3.20, p=1
  K=11, σ=3.20, p=1: Calibration Acc=1.0000, Brier=0.0408
Training Weighted KNN 113/160: k=11, σ=3.30, p=1
  K=11, σ=3.30, p=1: Calibration Acc=1.0000, Brier=0.0410
Training Weighted KNN 114/160: k=11, σ=3.40, p=1
  K=11, σ=3.40, p=1: Calibration Acc=1.0000, Brier=0.0411
Training Weighted KNN 115/160: k=11, σ=3.50, p=1
  K=11, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0412
Training Weighted KNN 116/160: k=11, σ=3.60, p=1
  K=11, σ=3.60, p=1: Calibration Acc=1.0000, Brier=0.0413
Training Weighted KNN 117/160: k=11, σ=3.70, p=1
  K=11, σ=3.70, p=1: Calibration Acc=1.0000, Brier=0.0414
Training Weighted KNN 118/160: k=11, σ=3.80, p=1
  K=11, σ=3.80, p=1: Calibration Acc=1.0000, Brier=0.0414
Training Weighted KNN 119/160: k=11, σ=3.90, p=1
  K=11, σ=3.90, p=1: Calibration Acc=1.0000, Brier=0.0415
Training Weighted KNN 120/160: k=11, σ=4.00, p=1
  K=11, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0416
Training Weighted KNN 121/160: k=13, σ=0.10, p=1
  K=13, σ=0.10, p=1: Calibration Acc=1.0000, Brier=0.0357
Training Weighted KNN 122/160: k=13, σ=0.20, p=1
  K=13, σ=0.20, p=1: Calibration Acc=1.0000, Brier=0.0357
Training Weighted KNN 123/160: k=13, σ=0.30, p=1
  K=13, σ=0.30, p=1: Calibration Acc=1.0000, Brier=0.0357
Training Weighted KNN 124/160: k=13, σ=0.40, p=1
  K=13, σ=0.40, p=1: Calibration Acc=1.0000, Brier=0.0358
Training Weighted KNN 125/160: k=13, σ=0.50, p=1
  K=13, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0359
Training Weighted KNN 126/160: k=13, σ=0.60, p=1
  K=13, σ=0.60, p=1: Calibration Acc=1.0000, Brier=0.0359
Training Weighted KNN 127/160: k=13, σ=0.70, p=1
  K=13, σ=0.70, p=1: Calibration Acc=1.0000, Brier=0.0360
Training Weighted KNN 128/160: k=13, σ=0.80, p=1
  K=13, σ=0.80, p=1: Calibration Acc=1.0000, Brier=0.0361
Training Weighted KNN 129/160: k=13, σ=0.90, p=1
  K=13, σ=0.90, p=1: Calibration Acc=1.0000, Brier=0.0362
Training Weighted KNN 130/160: k=13, σ=1.00, p=1
  K=13, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0364
Training Weighted KNN 131/160: k=13, σ=1.10, p=1
  K=13, σ=1.10, p=1: Calibration Acc=1.0000, Brier=0.0365
Training Weighted KNN 132/160: k=13, σ=1.20, p=1
  K=13, σ=1.20, p=1: Calibration Acc=1.0000, Brier=0.0366
Training Weighted KNN 133/160: k=13, σ=1.30, p=1
  K=13, σ=1.30, p=1: Calibration Acc=1.0000, Brier=0.0368
Training Weighted KNN 134/160: k=13, σ=1.40, p=1
  K=13, σ=1.40, p=1: Calibration Acc=1.0000, Brier=0.0369
Training Weighted KNN 135/160: k=13, σ=1.50, p=1
  K=13, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0371
Training Weighted KNN 136/160: k=13, σ=1.60, p=1
  K=13, σ=1.60, p=1: Calibration Acc=1.0000, Brier=0.0372
Training Weighted KNN 137/160: k=13, σ=1.70, p=1
  K=13, σ=1.70, p=1: Calibration Acc=1.0000, Brier=0.0374
Training Weighted KNN 138/160: k=13, σ=1.80, p=1
  K=13, σ=1.80, p=1: Calibration Acc=1.0000, Brier=0.0375
Training Weighted KNN 139/160: k=13, σ=1.90, p=1
  K=13, σ=1.90, p=1: Calibration Acc=1.0000, Brier=0.0377
Training Weighted KNN 140/160: k=13, σ=2.00, p=1
  K=13, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0378
Training Weighted KNN 141/160: k=13, σ=2.10, p=1
  K=13, σ=2.10, p=1: Calibration Acc=1.0000, Brier=0.0380
Training Weighted KNN 142/160: k=13, σ=2.20, p=1
  K=13, σ=2.20, p=1: Calibration Acc=1.0000, Brier=0.0381
Training Weighted KNN 143/160: k=13, σ=2.30, p=1
  K=13, σ=2.30, p=1: Calibration Acc=1.0000, Brier=0.0383
Training Weighted KNN 144/160: k=13, σ=2.40, p=1
  K=13, σ=2.40, p=1: Calibration Acc=1.0000, Brier=0.0384
Training Weighted KNN 145/160: k=13, σ=2.50, p=1
  K=13, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0386
Training Weighted KNN 146/160: k=13, σ=2.60, p=1
  K=13, σ=2.60, p=1: Calibration Acc=1.0000, Brier=0.0387
Training Weighted KNN 147/160: k=13, σ=2.70, p=1
  K=13, σ=2.70, p=1: Calibration Acc=1.0000, Brier=0.0389
Training Weighted KNN 148/160: k=13, σ=2.80, p=1
  K=13, σ=2.80, p=1: Calibration Acc=1.0000, Brier=0.0390
Training Weighted KNN 149/160: k=13, σ=2.90, p=1
  K=13, σ=2.90, p=1: Calibration Acc=1.0000, Brier=0.0391
Training Weighted KNN 150/160: k=13, σ=3.00, p=1
  K=13, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0393
Training Weighted KNN 151/160: k=13, σ=3.10, p=1
  K=13, σ=3.10, p=1: Calibration Acc=1.0000, Brier=0.0394
Training Weighted KNN 152/160: k=13, σ=3.20, p=1
  K=13, σ=3.20, p=1: Calibration Acc=1.0000, Brier=0.0395
Training Weighted KNN 153/160: k=13, σ=3.30, p=1
  K=13, σ=3.30, p=1: Calibration Acc=1.0000, Brier=0.0396
Training Weighted KNN 154/160: k=13, σ=3.40, p=1
  K=13, σ=3.40, p=1: Calibration Acc=1.0000, Brier=0.0397
Training Weighted KNN 155/160: k=13, σ=3.50, p=1
  K=13, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0398
Training Weighted KNN 156/160: k=13, σ=3.60, p=1
  K=13, σ=3.60, p=1: Calibration Acc=1.0000, Brier=0.0399
Training Weighted KNN 157/160: k=13, σ=3.70, p=1
  K=13, σ=3.70, p=1: Calibration Acc=1.0000, Brier=0.0400
Training Weighted KNN 158/160: k=13, σ=3.80, p=1
  K=13, σ=3.80, p=1: Calibration Acc=1.0000, Brier=0.0401
Training Weighted KNN 159/160: k=13, σ=3.90, p=1
  K=13, σ=3.90, p=1: Calibration Acc=1.0000, Brier=0.0402
Training Weighted KNN 160/160: k=13, σ=4.00, p=1
  K=13, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0402

=== Training Complete ===
Process started with PID: 3149
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (10, 2)
Calibration labels shape: (10,)
Calibration class distribution: {1: 10}
Minimum samples per class: 10

Class 1 calibration scores (10 points):
  Min: 0.6104
  Max: 1.0000
  Mean: 0.8585
  Std: 0.1300
  Percentiles: 25%=0.7896, 50%=0.8265, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {1: 10}

Export: Class 1 calibration scores (10 points):
  Min: 0.6104
  Max: 1.0000
  Mean: 0.8585
  Std: 0.1300
  Percentiles: 25%=0.7896, 50%=0.8265, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity
TEST CALLBACK TRIGGERED: n_clicks=None
measured_levels shape: (66,)
measured_class_labels shape: (66,)
measured_indices shape: (66,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 65
measured_indices range: 0 to 277
Level 0: level_mask shape: (66,), sum: 29
Level 1: level_mask shape: (66,), sum: 27
Level 2: level_mask shape: (66,), sum: 10

Starting Weighted KNN training process... using training dataset (56 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 56
Class distribution in training data: {0: 14, 1: 42}
K Min: 3
K Max: 10
K Step: 1
σ Min: 0.5
σ Max: 2
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 10 points
Training set class distribution: {0: 14, 1: 42}
Calibration set class distribution: {1: 10}
Training Weighted KNN 1/32: k=3, σ=0.50, p=1
  K=3, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.0593
Training Weighted KNN 2/32: k=3, σ=1.00, p=1
  K=3, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.0604
Training Weighted KNN 3/32: k=3, σ=1.50, p=1
  K=3, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.0622
Training Weighted KNN 4/32: k=3, σ=2.00, p=1
  K=3, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.0642
Training Weighted KNN 5/32: k=4, σ=0.50, p=1
  K=4, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.0511
Training Weighted KNN 6/32: k=4, σ=1.00, p=1
  K=4, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.0520
Training Weighted KNN 7/32: k=4, σ=1.50, p=1
  K=4, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.0533
Training Weighted KNN 8/32: k=4, σ=2.00, p=1
  K=4, σ=2.00, p=1: Calibration Acc=0.9000, Brier=0.0549
Training Weighted KNN 9/32: k=5, σ=0.50, p=1
  K=5, σ=0.50, p=1: Calibration Acc=0.9000, Brier=0.0489
Training Weighted KNN 10/32: k=5, σ=1.00, p=1
  K=5, σ=1.00, p=1: Calibration Acc=0.9000, Brier=0.0498
Training Weighted KNN 11/32: k=5, σ=1.50, p=1
  K=5, σ=1.50, p=1: Calibration Acc=0.9000, Brier=0.0512
Training Weighted KNN 12/32: k=5, σ=2.00, p=1
  K=5, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0528
Training Weighted KNN 13/32: k=6, σ=0.50, p=1
  K=6, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0451
Training Weighted KNN 14/32: k=6, σ=1.00, p=1
  K=6, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0459
Training Weighted KNN 15/32: k=6, σ=1.50, p=1
  K=6, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0470
Training Weighted KNN 16/32: k=6, σ=2.00, p=1
  K=6, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0484
Training Weighted KNN 17/32: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0423
Training Weighted KNN 18/32: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0430
Training Weighted KNN 19/32: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0439
Training Weighted KNN 20/32: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0450
Training Weighted KNN 21/32: k=8, σ=0.50, p=1
  K=8, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0400
Training Weighted KNN 22/32: k=8, σ=1.00, p=1
  K=8, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0406
Training Weighted KNN 23/32: k=8, σ=1.50, p=1
  K=8, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0414
Training Weighted KNN 24/32: k=8, σ=2.00, p=1
  K=8, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0424
Training Weighted KNN 25/32: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0406
Training Weighted KNN 26/32: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0413
Training Weighted KNN 27/32: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0424
Training Weighted KNN 28/32: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0436
Training Weighted KNN 29/32: k=10, σ=0.50, p=1
  K=10, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0383
Training Weighted KNN 30/32: k=10, σ=1.00, p=1
  K=10, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0388
Training Weighted KNN 31/32: k=10, σ=1.50, p=1
  K=10, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0396
Training Weighted KNN 32/32: k=10, σ=2.00, p=1
  K=10, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0406

=== Training Complete ===
Process started with PID: 4345
Processing training results

Starting Weighted KNN training process... using training dataset (36 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 36
Class distribution in training data: {0: 11, 1: 25}
K Min: 3
K Max: 10
K Step: 1
σ Min: 0.5
σ Max: 2
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 11, 1: 25}
Calibration set class distribution: {0: 3, 1: 17}
Training Weighted KNN 1/32: k=3, σ=0.50, p=1
  K=3, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.1326
Training Weighted KNN 2/32: k=3, σ=1.00, p=1
  K=3, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.1329
Training Weighted KNN 3/32: k=3, σ=1.50, p=1
  K=3, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.1334
Training Weighted KNN 4/32: k=3, σ=2.00, p=1
  K=3, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.1341
Training Weighted KNN 5/32: k=4, σ=0.50, p=1
  K=4, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.1403
Training Weighted KNN 6/32: k=4, σ=1.00, p=1
  K=4, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.1407
Training Weighted KNN 7/32: k=4, σ=1.50, p=1
  K=4, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.1414
Training Weighted KNN 8/32: k=4, σ=2.00, p=1
  K=4, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.1422
Training Weighted KNN 9/32: k=5, σ=0.50, p=1
  K=5, σ=0.50, p=1: Calibration Acc=0.8000, Brier=0.1347
Training Weighted KNN 10/32: k=5, σ=1.00, p=1
  K=5, σ=1.00, p=1: Calibration Acc=0.8000, Brier=0.1350
Training Weighted KNN 11/32: k=5, σ=1.50, p=1
  K=5, σ=1.50, p=1: Calibration Acc=0.8000, Brier=0.1356
Training Weighted KNN 12/32: k=5, σ=2.00, p=1
  K=5, σ=2.00, p=1: Calibration Acc=0.8000, Brier=0.1363
Training Weighted KNN 13/32: k=6, σ=0.50, p=1
  K=6, σ=0.50, p=1: Calibration Acc=0.8000, Brier=0.1231
Training Weighted KNN 14/32: k=6, σ=1.00, p=1
  K=6, σ=1.00, p=1: Calibration Acc=0.8000, Brier=0.1232
Training Weighted KNN 15/32: k=6, σ=1.50, p=1
  K=6, σ=1.50, p=1: Calibration Acc=0.8000, Brier=0.1234
Training Weighted KNN 16/32: k=6, σ=2.00, p=1
  K=6, σ=2.00, p=1: Calibration Acc=0.8000, Brier=0.1235
Training Weighted KNN 17/32: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=0.8000, Brier=0.1114
Training Weighted KNN 18/32: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=0.8000, Brier=0.1113
Training Weighted KNN 19/32: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=0.8000, Brier=0.1110
Training Weighted KNN 20/32: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=0.8500, Brier=0.1107
Training Weighted KNN 21/32: k=8, σ=0.50, p=1
  K=8, σ=0.50, p=1: Calibration Acc=0.8500, Brier=0.1068
Training Weighted KNN 22/32: k=8, σ=1.00, p=1
  K=8, σ=1.00, p=1: Calibration Acc=0.8500, Brier=0.1066
Training Weighted KNN 23/32: k=8, σ=1.50, p=1
  K=8, σ=1.50, p=1: Calibration Acc=0.8500, Brier=0.1062
Training Weighted KNN 24/32: k=8, σ=2.00, p=1
  K=8, σ=2.00, p=1: Calibration Acc=0.8500, Brier=0.1058
Training Weighted KNN 25/32: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=0.8000, Brier=0.1073
Training Weighted KNN 26/32: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=0.8000, Brier=0.1071
Training Weighted KNN 27/32: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=0.8000, Brier=0.1067
Training Weighted KNN 28/32: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=0.8500, Brier=0.1063
Training Weighted KNN 29/32: k=10, σ=0.50, p=1
  K=10, σ=0.50, p=1: Calibration Acc=0.8500, Brier=0.1055
Training Weighted KNN 30/32: k=10, σ=1.00, p=1
  K=10, σ=1.00, p=1: Calibration Acc=0.8500, Brier=0.1053
Training Weighted KNN 31/32: k=10, σ=1.50, p=1
  K=10, σ=1.50, p=1: Calibration Acc=0.8500, Brier=0.1049
Training Weighted KNN 32/32: k=10, σ=2.00, p=1
  K=10, σ=2.00, p=1: Calibration Acc=0.8500, Brier=0.1045

=== Training Complete ===
Process started with PID: 4437
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 3, 1: 17}
Minimum samples per class: 3

Class 1 calibration scores (17 points):
  Min: 0.3512
  Max: 1.0000
  Mean: 0.8593
  Std: 0.2368
  Percentiles: 25%=0.7927, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (3 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.5357
  Std: 0.4114

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                 ●
                   ●
                     ●
                                ●
                                       ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.3512305454141706, 0.38796360438832445, 0.4272354217161527, 0.6487694545858296, 0.7926793789293232, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.6072030682374925, 1.0]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.0, 1: 0.3512305454141706}
  Class 1 threshold: 0.3512305454141706
  α=0.01: threshold=0.3512
  α=0.01: storing threshold=0.3512
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.0, 1: 0.3512305454141706}
  Class 1 threshold: 0.3512305454141706
  α=0.05: threshold=0.3512
  α=0.05: storing threshold=0.3512
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.0, 1: 0.38796360438832445}
  Class 1 threshold: 0.38796360438832445
  α=0.10: threshold=0.3880
  α=0.10: storing threshold=0.3880
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.0, 1: 0.6487694545858296}
  Class 1 threshold: 0.6487694545858296
  α=0.20: threshold=0.6488
  α=0.20: storing threshold=0.6488
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.0, 1: 0.7926793789293232}
  Class 1 threshold: 0.7926793789293232
  α=0.25: threshold=0.7927
  α=0.25: storing threshold=0.7927
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 3, 1: 17}
Minimum samples per class: 3

Class 1 calibration scores (17 points):
  Min: 0.3520
  Max: 1.0000
  Mean: 0.8596
  Std: 0.2365
  Percentiles: 25%=0.7963, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (3 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.5351
  Std: 0.4113

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                 ●
                   ●
                     ●
                                ●
                                       ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.3519647096580766, 0.38846421721306934, 0.42868447022133765, 0.6480352903419235, 0.7962879410228281, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.6054148536017472, 1.0]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.0, 1: 0.3519647096580766}
  Class 1 threshold: 0.3519647096580766
  α=0.01: threshold=0.3520
  α=0.01: storing threshold=0.3520
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.0, 1: 0.3519647096580766}
  Class 1 threshold: 0.3519647096580766
  α=0.05: threshold=0.3520
  α=0.05: storing threshold=0.3520
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.0, 1: 0.38846421721306934}
  Class 1 threshold: 0.38846421721306934
  α=0.10: threshold=0.3885
  α=0.10: storing threshold=0.3885
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.0, 1: 0.6480352903419235}
  Class 1 threshold: 0.6480352903419235
  α=0.20: threshold=0.6480
  α=0.20: storing threshold=0.6480
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.0, 1: 0.7962879410228281}
  Class 1 threshold: 0.7962879410228281
  α=0.25: threshold=0.7963
  α=0.25: storing threshold=0.7963
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 3, 1: 17}
Minimum samples per class: 3

Class 1 calibration scores (17 points):
  Min: 0.3512
  Max: 1.0000
  Mean: 0.8593
  Std: 0.2368
  Percentiles: 25%=0.7927, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (3 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.5357
  Std: 0.4114

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                 ●
                   ●
                     ●
                                ●
                                       ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.3512305454141706, 0.38796360438832445, 0.4272354217161527, 0.6487694545858296, 0.7926793789293232, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.6072030682374925, 1.0]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.0, 1: 0.3512305454141706}
  Class 1 threshold: 0.3512305454141706
  α=0.01: threshold=0.3512
  α=0.01: storing threshold=0.3512
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.0, 1: 0.3512305454141706}
  Class 1 threshold: 0.3512305454141706
  α=0.05: threshold=0.3512
  α=0.05: storing threshold=0.3512
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.0, 1: 0.38796360438832445}
  Class 1 threshold: 0.38796360438832445
  α=0.10: threshold=0.3880
  α=0.10: storing threshold=0.3880
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.0, 1: 0.6487694545858296}
  Class 1 threshold: 0.6487694545858296
  α=0.20: threshold=0.6488
  α=0.20: storing threshold=0.6488
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.0, 1: 0.7926793789293232}
  Class 1 threshold: 0.7926793789293232
  α=0.25: threshold=0.7927
  α=0.25: storing threshold=0.7927
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {0: 3, 1: 17}

Export: Class 1 calibration scores (17 points):
  Min: 0.3512
  Max: 1.0000
  Mean: 0.8593
  Std: 0.2368
  Percentiles: 25%=0.7927, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (3 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.5357
  Std: 0.4114

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
                 ●
                   ●
                     ●
                                ●
                                       ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.3512305454141706, 0.38796360438832445, 0.4272354217161527, 0.6487694545858296, 0.7926793789293232, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.0, 0.6072030682374925, 1.0]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.3512305454141706
Export: α=0.01: threshold=0.3512
Export: α=0.01: storing threshold=0.3512
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.3512305454141706
Export: α=0.05: threshold=0.3512
Export: α=0.05: storing threshold=0.3512
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.38796360438832445
Export: α=0.10: threshold=0.3880
Export: α=0.10: storing threshold=0.3880
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.6487694545858296
Export: α=0.20: threshold=0.6488
Export: α=0.20: storing threshold=0.6488
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.7926793789293232
Export: α=0.25: threshold=0.7927
Export: α=0.25: storing threshold=0.7927

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 180.0+ bytes
None

Existing table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m1        0.566558  ...        0.569419        0.569419

[1 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
         model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_lipid3        0.351231  ...        0.648769        0.792679

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      2 non-null      object 
 1   threshold_0.01  2 non-null      float64
 2   threshold_0.05  2 non-null      float64
 3   threshold_0.10  2 non-null      float64
 4   threshold_0.20  2 non-null      float64
 5   threshold_0.25  2 non-null      float64
dtypes: float64(5), object(1)
memory usage: 228.0+ bytes
None

Merged table content:
         model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0      knn_model_m1        0.566558  ...        0.569419        0.569419
1  knn_model_lipid3        0.351231  ...        0.648769        0.792679

[2 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'
measured_levels shape: (126,)
measured_class_labels shape: (126,)
measured_indices shape: (126,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (126,), sum: 29
Level 1: level_mask shape: (126,), sum: 28
Level 2: level_mask shape: (126,), sum: 49
Level 3: level_mask shape: (126,), sum: 20

Starting Weighted KNN training process... using training dataset (86 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 86
Class distribution in training data: {0: 75, 1: 11}
K Min: 3
K Max: 10
K Step: 1
σ Min: 0.5
σ Max: 2
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 75, 1: 11}
Calibration set class distribution: {0: 15, 1: 5}
Training Weighted KNN 1/32: k=3, σ=0.50, p=1
  K=3, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2525
Training Weighted KNN 2/32: k=3, σ=1.00, p=1
  K=3, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2521
Training Weighted KNN 3/32: k=3, σ=1.50, p=1
  K=3, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2517
Training Weighted KNN 4/32: k=3, σ=2.00, p=1
  K=3, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2513
Training Weighted KNN 5/32: k=4, σ=0.50, p=1
  K=4, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2392
Training Weighted KNN 6/32: k=4, σ=1.00, p=1
  K=4, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2389
Training Weighted KNN 7/32: k=4, σ=1.50, p=1
  K=4, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2386
Training Weighted KNN 8/32: k=4, σ=2.00, p=1
  K=4, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2383
Training Weighted KNN 9/32: k=5, σ=0.50, p=1
  K=5, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2522
Training Weighted KNN 10/32: k=5, σ=1.00, p=1
  K=5, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2523
Training Weighted KNN 11/32: k=5, σ=1.50, p=1
  K=5, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2526
Training Weighted KNN 12/32: k=5, σ=2.00, p=1
  K=5, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2530
Training Weighted KNN 13/32: k=6, σ=0.50, p=1
  K=6, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2500
Training Weighted KNN 14/32: k=6, σ=1.00, p=1
  K=6, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2500
Training Weighted KNN 15/32: k=6, σ=1.50, p=1
  K=6, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2502
Training Weighted KNN 16/32: k=6, σ=2.00, p=1
  K=6, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2504
Training Weighted KNN 17/32: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2533
Training Weighted KNN 18/32: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2534
Training Weighted KNN 19/32: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2536
Training Weighted KNN 20/32: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2540
Training Weighted KNN 21/32: k=8, σ=0.50, p=1
  K=8, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2479
Training Weighted KNN 22/32: k=8, σ=1.00, p=1
  K=8, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2478
Training Weighted KNN 23/32: k=8, σ=1.50, p=1
  K=8, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2477
Training Weighted KNN 24/32: k=8, σ=2.00, p=1
  K=8, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2476
Training Weighted KNN 25/32: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2456
Training Weighted KNN 26/32: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2454
Training Weighted KNN 27/32: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2452
Training Weighted KNN 28/32: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2449
Training Weighted KNN 29/32: k=10, σ=0.50, p=1
  K=10, σ=0.50, p=1: Calibration Acc=0.7500, Brier=0.2404
Training Weighted KNN 30/32: k=10, σ=1.00, p=1
  K=10, σ=1.00, p=1: Calibration Acc=0.7500, Brier=0.2400
Training Weighted KNN 31/32: k=10, σ=1.50, p=1
  K=10, σ=1.50, p=1: Calibration Acc=0.7500, Brier=0.2395
Training Weighted KNN 32/32: k=10, σ=2.00, p=1
  K=10, σ=2.00, p=1: Calibration Acc=0.7500, Brier=0.2389

=== Training Complete ===
Process started with PID: 8327
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 15, 1: 5}
Minimum samples per class: 5

Class 1 calibration scores (5 points):
  Min: 0.0000
  Max: 0.5742
  Mean: 0.1291
  Std: 0.2232
  Percentiles: 25%=0.0000, 50%=0.0326, 75%=0.0385, 95%=0.4671, 99%=0.5528

Class 0 calibration scores (15 points):
  Min: 0.0000
  Max: 0.6273
  Mean: 0.1271
  Std: 0.1869

Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
●
 ●
 ●
                            ●
0.0==================================================1.0

Class 1 scores (sorted): [0.0, 0.0, 0.032567100067283535, 0.03845795188589105, 0.5742299843856636]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044258143103918075, 0.07662974101467353, 0.1460284455837312, 0.31051185173262114, 0.32786169005460974, 0.37346866807043083, 0.6273353808379833]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.37266461916201665, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.01: threshold=0.0000
  α=0.01: storing threshold=0.0000
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.37266461916201665, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.05: threshold=0.0000
  α=0.05: storing threshold=0.0000
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.6265313319295692, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.10: threshold=0.0000
  α=0.10: storing threshold=0.0000
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.6721383099453904, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.20: threshold=0.0000
  α=0.20: storing threshold=0.0000
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.6894881482673789, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.25: threshold=0.0000
  α=0.25: storing threshold=0.0000
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {0: 15, 1: 5}

Export: Class 1 calibration scores (5 points):
  Min: 0.0000
  Max: 0.5742
  Mean: 0.1291
  Std: 0.2232
  Percentiles: 25%=0.0000, 50%=0.0326, 75%=0.0385, 95%=0.4671, 99%=0.5528

Export: Class 0 calibration scores (15 points):
  Min: 0.0000
  Max: 0.6273
  Mean: 0.1271
  Std: 0.1869

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
●
 ●
 ●
                            ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.0, 0.0, 0.032567100067283535, 0.03845795188589105, 0.5742299843856636]

Export: Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044258143103918075, 0.07662974101467353, 0.1460284455837312, 0.31051185173262114, 0.32786169005460974, 0.37346866807043083, 0.6273353808379833]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.0
Export: α=0.01: threshold=0.0000
Export: α=0.01: storing threshold=0.0000
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.0
Export: α=0.05: threshold=0.0000
Export: α=0.05: storing threshold=0.0000
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.0
Export: α=0.10: threshold=0.0000
Export: α=0.10: storing threshold=0.0000
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.0
Export: α=0.20: threshold=0.0000
Export: α=0.20: storing threshold=0.0000
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.0
Export: α=0.25: threshold=0.0000
Export: α=0.25: storing threshold=0.0000

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      2 non-null      object 
 1   threshold_0.01  2 non-null      float64
 2   threshold_0.05  2 non-null      float64
 3   threshold_0.10  2 non-null      float64
 4   threshold_0.20  2 non-null      float64
 5   threshold_0.25  2 non-null      float64
dtypes: float64(5), object(1)
memory usage: 228.0+ bytes
None

Existing table content:
         model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0      knn_model_m1        0.566558  ...        0.569419        0.569419
1  knn_model_lipid3        0.351231  ...        0.648769        0.792679

[2 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
           model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_mystery1             0.0  ...             0.0             0.0

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      3 non-null      object 
 1   threshold_0.01  3 non-null      float64
 2   threshold_0.05  3 non-null      float64
 3   threshold_0.10  3 non-null      float64
 4   threshold_0.20  3 non-null      float64
 5   threshold_0.25  3 non-null      float64
dtypes: float64(5), object(1)
memory usage: 276.0+ bytes
None

Merged table content:
           model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0        knn_model_m1        0.566558  ...        0.569419        0.569419
1    knn_model_lipid3        0.351231  ...        0.648769        0.792679
2  knn_model_mystery1        0.000000  ...        0.000000        0.000000

[3 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'

Starting Weighted KNN training process... using training dataset (57 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 57
Class distribution in training data: {0: 50, 1: 7}
K Min: 3
K Max: 10
K Step: 1
σ Min: 0.5
σ Max: 2
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 49 points
Training set class distribution: {0: 50, 1: 7}
Calibration set class distribution: {0: 40, 1: 9}
Training Weighted KNN 1/32: k=3, σ=0.50, p=1
  K=3, σ=0.50, p=1: Calibration Acc=0.7959, Brier=0.1554
Training Weighted KNN 2/32: k=3, σ=1.00, p=1
  K=3, σ=1.00, p=1: Calibration Acc=0.7959, Brier=0.1552
Training Weighted KNN 3/32: k=3, σ=1.50, p=1
  K=3, σ=1.50, p=1: Calibration Acc=0.7959, Brier=0.1550
Training Weighted KNN 4/32: k=3, σ=2.00, p=1
  K=3, σ=2.00, p=1: Calibration Acc=0.7959, Brier=0.1548
Training Weighted KNN 5/32: k=4, σ=0.50, p=1
  K=4, σ=0.50, p=1: Calibration Acc=0.7551, Brier=0.1437
Training Weighted KNN 6/32: k=4, σ=1.00, p=1
  K=4, σ=1.00, p=1: Calibration Acc=0.7755, Brier=0.1432
Training Weighted KNN 7/32: k=4, σ=1.50, p=1
  K=4, σ=1.50, p=1: Calibration Acc=0.7755, Brier=0.1425
Training Weighted KNN 8/32: k=4, σ=2.00, p=1
  K=4, σ=2.00, p=1: Calibration Acc=0.7755, Brier=0.1417
Training Weighted KNN 9/32: k=5, σ=0.50, p=1
  K=5, σ=0.50, p=1: Calibration Acc=0.8163, Brier=0.1415
Training Weighted KNN 10/32: k=5, σ=1.00, p=1
  K=5, σ=1.00, p=1: Calibration Acc=0.8163, Brier=0.1410
Training Weighted KNN 11/32: k=5, σ=1.50, p=1
  K=5, σ=1.50, p=1: Calibration Acc=0.8163, Brier=0.1402
Training Weighted KNN 12/32: k=5, σ=2.00, p=1
  K=5, σ=2.00, p=1: Calibration Acc=0.8367, Brier=0.1393
Training Weighted KNN 13/32: k=6, σ=0.50, p=1
  K=6, σ=0.50, p=1: Calibration Acc=0.7755, Brier=0.1418
Training Weighted KNN 14/32: k=6, σ=1.00, p=1
  K=6, σ=1.00, p=1: Calibration Acc=0.7755, Brier=0.1414
Training Weighted KNN 15/32: k=6, σ=1.50, p=1
  K=6, σ=1.50, p=1: Calibration Acc=0.7755, Brier=0.1408
Training Weighted KNN 16/32: k=6, σ=2.00, p=1
  K=6, σ=2.00, p=1: Calibration Acc=0.7959, Brier=0.1402
Training Weighted KNN 17/32: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=0.7959, Brier=0.1420
Training Weighted KNN 18/32: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=0.7959, Brier=0.1417
Training Weighted KNN 19/32: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=0.7959, Brier=0.1412
Training Weighted KNN 20/32: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=0.7959, Brier=0.1408
Training Weighted KNN 21/32: k=8, σ=0.50, p=1
  K=8, σ=0.50, p=1: Calibration Acc=0.7959, Brier=0.1398
Training Weighted KNN 22/32: k=8, σ=1.00, p=1
  K=8, σ=1.00, p=1: Calibration Acc=0.7959, Brier=0.1394
Training Weighted KNN 23/32: k=8, σ=1.50, p=1
  K=8, σ=1.50, p=1: Calibration Acc=0.7959, Brier=0.1389
Training Weighted KNN 24/32: k=8, σ=2.00, p=1
  K=8, σ=2.00, p=1: Calibration Acc=0.8163, Brier=0.1385
Training Weighted KNN 25/32: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=0.8163, Brier=0.1384
Training Weighted KNN 26/32: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=0.8163, Brier=0.1380
Training Weighted KNN 27/32: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=0.8367, Brier=0.1375
Training Weighted KNN 28/32: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=0.8367, Brier=0.1370
Training Weighted KNN 29/32: k=10, σ=0.50, p=1
  K=10, σ=0.50, p=1: Calibration Acc=0.8163, Brier=0.1378
Training Weighted KNN 30/32: k=10, σ=1.00, p=1
  K=10, σ=1.00, p=1: Calibration Acc=0.8163, Brier=0.1374
Training Weighted KNN 31/32: k=10, σ=1.50, p=1
  K=10, σ=1.50, p=1: Calibration Acc=0.8163, Brier=0.1370
Training Weighted KNN 32/32: k=10, σ=2.00, p=1
  K=10, σ=2.00, p=1: Calibration Acc=0.8367, Brier=0.1365

=== Training Complete ===
Process started with PID: 8472
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (49, 2)
Calibration labels shape: (49,)
Calibration class distribution: {0: 40, 1: 9}
Minimum samples per class: 9

Class 1 calibration scores (9 points):
  Min: 0.0000
  Max: 0.6243
  Mean: 0.2877
  Std: 0.2497
  Percentiles: 25%=0.0359, 50%=0.3754, 75%=0.4613, 95%=0.6169, 99%=0.6228

Class 0 calibration scores (40 points):
  Min: 0.0000
  Max: 0.5615
  Mean: 0.1121
  Std: 0.1661

Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
●
 ●
  ●
                  ●
                     ●
                       ●
                              ●
                               ●
0.0==================================================1.0

Class 1 scores (sorted): [0.0, 0.0, 0.035919324161540055, 0.048116981736502445, 0.3754177682210767, 0.43818773634831365, 0.4612731279008409, 0.605880724725914, 0.6242977890855489]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03323631244275649, 0.03640471298772052, 0.03799180557886699, 0.04053540315009619, 0.04231737960521537, 0.04300498246519744, 0.04534583385776583, 0.05557877801231805, 0.05585198279094301, 0.05804727345084511, 0.06121873119088087, 0.10528338325454312, 0.167677358942539, 0.1695409768098591, 0.2067105200577776, 0.22910123941681898, 0.320147822576653, 0.3308191791798083, 0.43123673961831044, 0.47091078655669294, 0.47433984653731287, 0.5091095721671168, 0.5614888802355328]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.43851111976446727, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.01: threshold=0.0000
  α=0.01: storing threshold=0.0000
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.49089042783288306, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.05: threshold=0.0000
  α=0.05: storing threshold=0.0000
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.5290892134433071, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.10: threshold=0.0000
  α=0.10: storing threshold=0.0000
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.7708987605831809, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.20: threshold=0.0000
  α=0.20: storing threshold=0.0000
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.8304590231901409, 1: 0.035919324161540055}
  Class 1 threshold: 0.035919324161540055
  α=0.25: threshold=0.0359
  α=0.25: storing threshold=0.0359
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {0: 40, 1: 9}

Export: Class 1 calibration scores (9 points):
  Min: 0.0000
  Max: 0.6243
  Mean: 0.2877
  Std: 0.2497
  Percentiles: 25%=0.0359, 50%=0.3754, 75%=0.4613, 95%=0.6169, 99%=0.6228

Export: Class 0 calibration scores (40 points):
  Min: 0.0000
  Max: 0.5615
  Mean: 0.1121
  Std: 0.1661

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
●
 ●
  ●
                  ●
                     ●
                       ●
                              ●
                               ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.0, 0.0, 0.035919324161540055, 0.048116981736502445, 0.3754177682210767, 0.43818773634831365, 0.4612731279008409, 0.605880724725914, 0.6242977890855489]

Export: Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03323631244275649, 0.03640471298772052, 0.03799180557886699, 0.04053540315009619, 0.04231737960521537, 0.04300498246519744, 0.04534583385776583, 0.05557877801231805, 0.05585198279094301, 0.05804727345084511, 0.06121873119088087, 0.10528338325454312, 0.167677358942539, 0.1695409768098591, 0.2067105200577776, 0.22910123941681898, 0.320147822576653, 0.3308191791798083, 0.43123673961831044, 0.47091078655669294, 0.47433984653731287, 0.5091095721671168, 0.5614888802355328]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.0
Export: α=0.01: threshold=0.0000
Export: α=0.01: storing threshold=0.0000
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.0
Export: α=0.05: threshold=0.0000
Export: α=0.05: storing threshold=0.0000
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.0
Export: α=0.10: threshold=0.0000
Export: α=0.10: storing threshold=0.0000
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.0
Export: α=0.20: threshold=0.0000
Export: α=0.20: storing threshold=0.0000
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.035919324161540055
Export: α=0.25: threshold=0.0359
Export: α=0.25: storing threshold=0.0359

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      3 non-null      object 
 1   threshold_0.01  3 non-null      float64
 2   threshold_0.05  3 non-null      float64
 3   threshold_0.10  3 non-null      float64
 4   threshold_0.20  3 non-null      float64
 5   threshold_0.25  3 non-null      float64
dtypes: float64(5), object(1)
memory usage: 276.0+ bytes
None

Existing table content:
           model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0        knn_model_m1        0.566558  ...        0.569419        0.569419
1    knn_model_lipid3        0.351231  ...        0.648769        0.792679
2  knn_model_mystery1        0.000000  ...        0.000000        0.000000

[3 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
           model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_mystery2             0.0  ...             0.0        0.035919

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      4 non-null      object 
 1   threshold_0.01  4 non-null      float64
 2   threshold_0.05  4 non-null      float64
 3   threshold_0.10  4 non-null      float64
 4   threshold_0.20  4 non-null      float64
 5   threshold_0.25  4 non-null      float64
dtypes: float64(5), object(1)
memory usage: 324.0+ bytes
None

Merged table content:
           model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0        knn_model_m1        0.566558  ...        0.569419        0.569419
1    knn_model_lipid3        0.351231  ...        0.648769        0.792679
2  knn_model_mystery1        0.000000  ...        0.000000        0.000000
3  knn_model_mystery2        0.000000  ...        0.000000        0.035919

[4 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'
measured_levels shape: (126,)
measured_class_labels shape: (126,)
measured_indices shape: (126,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (126,), sum: 29
Level 1: level_mask shape: (126,), sum: 28
Level 2: level_mask shape: (126,), sum: 49
Level 3: level_mask shape: (126,), sum: 20

Starting Weighted KNN training process... using training dataset (106 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 106
Class distribution in training data: {0: 72, 1: 34}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.1
σ Max: 5
σ Step: 0.1
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 72, 1: 34}
Calibration set class distribution: {0: 12, 1: 8}
Training Weighted KNN 1/150: k=7, σ=0.10, p=2
  K=7, σ=0.10, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 2/150: k=7, σ=0.20, p=2
  K=7, σ=0.20, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 3/150: k=7, σ=0.30, p=2
  K=7, σ=0.30, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 4/150: k=7, σ=0.40, p=2
  K=7, σ=0.40, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 5/150: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 6/150: k=7, σ=0.60, p=2
  K=7, σ=0.60, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 7/150: k=7, σ=0.70, p=2
  K=7, σ=0.70, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 8/150: k=7, σ=0.80, p=2
  K=7, σ=0.80, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 9/150: k=7, σ=0.90, p=2
  K=7, σ=0.90, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 10/150: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 11/150: k=7, σ=1.10, p=2
  K=7, σ=1.10, p=2: Calibration Acc=0.9500, Brier=0.0610
Training Weighted KNN 12/150: k=7, σ=1.20, p=2
  K=7, σ=1.20, p=2: Calibration Acc=0.9500, Brier=0.0611
Training Weighted KNN 13/150: k=7, σ=1.30, p=2
  K=7, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0611
Training Weighted KNN 14/150: k=7, σ=1.40, p=2
  K=7, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0612
Training Weighted KNN 15/150: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0613
Training Weighted KNN 16/150: k=7, σ=1.60, p=2
  K=7, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0616
Training Weighted KNN 17/150: k=7, σ=1.70, p=2
  K=7, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0618
Training Weighted KNN 18/150: k=7, σ=1.80, p=2
  K=7, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0622
Training Weighted KNN 19/150: k=7, σ=1.90, p=2
  K=7, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0627
Training Weighted KNN 20/150: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0632
Training Weighted KNN 21/150: k=7, σ=2.10, p=2
  K=7, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0639
Training Weighted KNN 22/150: k=7, σ=2.20, p=2
  K=7, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0647
Training Weighted KNN 23/150: k=7, σ=2.30, p=2
  K=7, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0655
Training Weighted KNN 24/150: k=7, σ=2.40, p=2
  K=7, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0665
Training Weighted KNN 25/150: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0675
Training Weighted KNN 26/150: k=7, σ=2.60, p=2
  K=7, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.0686
Training Weighted KNN 27/150: k=7, σ=2.70, p=2
  K=7, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.0697
Training Weighted KNN 28/150: k=7, σ=2.80, p=2
  K=7, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.0709
Training Weighted KNN 29/150: k=7, σ=2.90, p=2
  K=7, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.0722
Training Weighted KNN 30/150: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0734
Training Weighted KNN 31/150: k=7, σ=3.10, p=2
  K=7, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.0747
Training Weighted KNN 32/150: k=7, σ=3.20, p=2
  K=7, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.0759
Training Weighted KNN 33/150: k=7, σ=3.30, p=2
  K=7, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.0772
Training Weighted KNN 34/150: k=7, σ=3.40, p=2
  K=7, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.0784
Training Weighted KNN 35/150: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.0797
Training Weighted KNN 36/150: k=7, σ=3.60, p=2
  K=7, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.0808
Training Weighted KNN 37/150: k=7, σ=3.70, p=2
  K=7, σ=3.70, p=2: Calibration Acc=0.9000, Brier=0.0820
Training Weighted KNN 38/150: k=7, σ=3.80, p=2
  K=7, σ=3.80, p=2: Calibration Acc=0.9000, Brier=0.0831
Training Weighted KNN 39/150: k=7, σ=3.90, p=2
  K=7, σ=3.90, p=2: Calibration Acc=0.9000, Brier=0.0842
Training Weighted KNN 40/150: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.0853
Training Weighted KNN 41/150: k=7, σ=4.10, p=2
  K=7, σ=4.10, p=2: Calibration Acc=0.9000, Brier=0.0863
Training Weighted KNN 42/150: k=7, σ=4.20, p=2
  K=7, σ=4.20, p=2: Calibration Acc=0.9000, Brier=0.0872
Training Weighted KNN 43/150: k=7, σ=4.30, p=2
  K=7, σ=4.30, p=2: Calibration Acc=0.9000, Brier=0.0881
Training Weighted KNN 44/150: k=7, σ=4.40, p=2
  K=7, σ=4.40, p=2: Calibration Acc=0.9000, Brier=0.0890
Training Weighted KNN 45/150: k=7, σ=4.50, p=2
  K=7, σ=4.50, p=2: Calibration Acc=0.9000, Brier=0.0898
Training Weighted KNN 46/150: k=7, σ=4.60, p=2
  K=7, σ=4.60, p=2: Calibration Acc=0.8500, Brier=0.0906
Training Weighted KNN 47/150: k=7, σ=4.70, p=2
  K=7, σ=4.70, p=2: Calibration Acc=0.8500, Brier=0.0914
Training Weighted KNN 48/150: k=7, σ=4.80, p=2
  K=7, σ=4.80, p=2: Calibration Acc=0.8500, Brier=0.0921
Training Weighted KNN 49/150: k=7, σ=4.90, p=2
  K=7, σ=4.90, p=2: Calibration Acc=0.8500, Brier=0.0928
Training Weighted KNN 50/150: k=7, σ=5.00, p=2
  K=7, σ=5.00, p=2: Calibration Acc=0.8500, Brier=0.0934
Training Weighted KNN 51/150: k=9, σ=0.10, p=2
  K=9, σ=0.10, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 52/150: k=9, σ=0.20, p=2
  K=9, σ=0.20, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 53/150: k=9, σ=0.30, p=2
  K=9, σ=0.30, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 54/150: k=9, σ=0.40, p=2
  K=9, σ=0.40, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 55/150: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 56/150: k=9, σ=0.60, p=2
  K=9, σ=0.60, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 57/150: k=9, σ=0.70, p=2
  K=9, σ=0.70, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 58/150: k=9, σ=0.80, p=2
  K=9, σ=0.80, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 59/150: k=9, σ=0.90, p=2
  K=9, σ=0.90, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 60/150: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 61/150: k=9, σ=1.10, p=2
  K=9, σ=1.10, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 62/150: k=9, σ=1.20, p=2
  K=9, σ=1.20, p=2: Calibration Acc=0.9500, Brier=0.0604
Training Weighted KNN 63/150: k=9, σ=1.30, p=2
  K=9, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0605
Training Weighted KNN 64/150: k=9, σ=1.40, p=2
  K=9, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0606
Training Weighted KNN 65/150: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0607
Training Weighted KNN 66/150: k=9, σ=1.60, p=2
  K=9, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0609
Training Weighted KNN 67/150: k=9, σ=1.70, p=2
  K=9, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0612
Training Weighted KNN 68/150: k=9, σ=1.80, p=2
  K=9, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0615
Training Weighted KNN 69/150: k=9, σ=1.90, p=2
  K=9, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0620
Training Weighted KNN 70/150: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0626
Training Weighted KNN 71/150: k=9, σ=2.10, p=2
  K=9, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0632
Training Weighted KNN 72/150: k=9, σ=2.20, p=2
  K=9, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0640
Training Weighted KNN 73/150: k=9, σ=2.30, p=2
  K=9, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0648
Training Weighted KNN 74/150: k=9, σ=2.40, p=2
  K=9, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0657
Training Weighted KNN 75/150: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0667
Training Weighted KNN 76/150: k=9, σ=2.60, p=2
  K=9, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.0678
Training Weighted KNN 77/150: k=9, σ=2.70, p=2
  K=9, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.0689
Training Weighted KNN 78/150: k=9, σ=2.80, p=2
  K=9, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.0701
Training Weighted KNN 79/150: k=9, σ=2.90, p=2
  K=9, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.0712
Training Weighted KNN 80/150: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0724
Training Weighted KNN 81/150: k=9, σ=3.10, p=2
  K=9, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.0736
Training Weighted KNN 82/150: k=9, σ=3.20, p=2
  K=9, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.0748
Training Weighted KNN 83/150: k=9, σ=3.30, p=2
  K=9, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.0759
Training Weighted KNN 84/150: k=9, σ=3.40, p=2
  K=9, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.0770
Training Weighted KNN 85/150: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.0781
Training Weighted KNN 86/150: k=9, σ=3.60, p=2
  K=9, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.0792
Training Weighted KNN 87/150: k=9, σ=3.70, p=2
  K=9, σ=3.70, p=2: Calibration Acc=0.9000, Brier=0.0802
Training Weighted KNN 88/150: k=9, σ=3.80, p=2
  K=9, σ=3.80, p=2: Calibration Acc=0.9000, Brier=0.0812
Training Weighted KNN 89/150: k=9, σ=3.90, p=2
  K=9, σ=3.90, p=2: Calibration Acc=0.9000, Brier=0.0821
Training Weighted KNN 90/150: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=0.9500, Brier=0.0830
Training Weighted KNN 91/150: k=9, σ=4.10, p=2
  K=9, σ=4.10, p=2: Calibration Acc=0.9500, Brier=0.0838
Training Weighted KNN 92/150: k=9, σ=4.20, p=2
  K=9, σ=4.20, p=2: Calibration Acc=0.9000, Brier=0.0845
Training Weighted KNN 93/150: k=9, σ=4.30, p=2
  K=9, σ=4.30, p=2: Calibration Acc=0.9000, Brier=0.0853
Training Weighted KNN 94/150: k=9, σ=4.40, p=2
  K=9, σ=4.40, p=2: Calibration Acc=0.9000, Brier=0.0860
Training Weighted KNN 95/150: k=9, σ=4.50, p=2
  K=9, σ=4.50, p=2: Calibration Acc=0.9000, Brier=0.0866
Training Weighted KNN 96/150: k=9, σ=4.60, p=2
  K=9, σ=4.60, p=2: Calibration Acc=0.9000, Brier=0.0872
Training Weighted KNN 97/150: k=9, σ=4.70, p=2
  K=9, σ=4.70, p=2: Calibration Acc=0.9000, Brier=0.0877
Training Weighted KNN 98/150: k=9, σ=4.80, p=2
  K=9, σ=4.80, p=2: Calibration Acc=0.9000, Brier=0.0882
Training Weighted KNN 99/150: k=9, σ=4.90, p=2
  K=9, σ=4.90, p=2: Calibration Acc=0.9000, Brier=0.0887
Training Weighted KNN 100/150: k=9, σ=5.00, p=2
  K=9, σ=5.00, p=2: Calibration Acc=0.9000, Brier=0.0891
Training Weighted KNN 101/150: k=11, σ=0.10, p=2
  K=11, σ=0.10, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 102/150: k=11, σ=0.20, p=2
  K=11, σ=0.20, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 103/150: k=11, σ=0.30, p=2
  K=11, σ=0.30, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 104/150: k=11, σ=0.40, p=2
  K=11, σ=0.40, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 105/150: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 106/150: k=11, σ=0.60, p=2
  K=11, σ=0.60, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 107/150: k=11, σ=0.70, p=2
  K=11, σ=0.70, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 108/150: k=11, σ=0.80, p=2
  K=11, σ=0.80, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 109/150: k=11, σ=0.90, p=2
  K=11, σ=0.90, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 110/150: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.9500, Brier=0.0607
Training Weighted KNN 111/150: k=11, σ=1.10, p=2
  K=11, σ=1.10, p=2: Calibration Acc=0.9500, Brier=0.0608
Training Weighted KNN 112/150: k=11, σ=1.20, p=2
  K=11, σ=1.20, p=2: Calibration Acc=0.9500, Brier=0.0608
Training Weighted KNN 113/150: k=11, σ=1.30, p=2
  K=11, σ=1.30, p=2: Calibration Acc=0.9000, Brier=0.0609
Training Weighted KNN 114/150: k=11, σ=1.40, p=2
  K=11, σ=1.40, p=2: Calibration Acc=0.9000, Brier=0.0610
Training Weighted KNN 115/150: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0612
Training Weighted KNN 116/150: k=11, σ=1.60, p=2
  K=11, σ=1.60, p=2: Calibration Acc=0.9000, Brier=0.0615
Training Weighted KNN 117/150: k=11, σ=1.70, p=2
  K=11, σ=1.70, p=2: Calibration Acc=0.9000, Brier=0.0618
Training Weighted KNN 118/150: k=11, σ=1.80, p=2
  K=11, σ=1.80, p=2: Calibration Acc=0.9000, Brier=0.0623
Training Weighted KNN 119/150: k=11, σ=1.90, p=2
  K=11, σ=1.90, p=2: Calibration Acc=0.9000, Brier=0.0629
Training Weighted KNN 120/150: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0635
Training Weighted KNN 121/150: k=11, σ=2.10, p=2
  K=11, σ=2.10, p=2: Calibration Acc=0.9000, Brier=0.0643
Training Weighted KNN 122/150: k=11, σ=2.20, p=2
  K=11, σ=2.20, p=2: Calibration Acc=0.9000, Brier=0.0653
Training Weighted KNN 123/150: k=11, σ=2.30, p=2
  K=11, σ=2.30, p=2: Calibration Acc=0.9000, Brier=0.0663
Training Weighted KNN 124/150: k=11, σ=2.40, p=2
  K=11, σ=2.40, p=2: Calibration Acc=0.9000, Brier=0.0674
Training Weighted KNN 125/150: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0686
Training Weighted KNN 126/150: k=11, σ=2.60, p=2
  K=11, σ=2.60, p=2: Calibration Acc=0.9000, Brier=0.0699
Training Weighted KNN 127/150: k=11, σ=2.70, p=2
  K=11, σ=2.70, p=2: Calibration Acc=0.9000, Brier=0.0713
Training Weighted KNN 128/150: k=11, σ=2.80, p=2
  K=11, σ=2.80, p=2: Calibration Acc=0.9000, Brier=0.0727
Training Weighted KNN 129/150: k=11, σ=2.90, p=2
  K=11, σ=2.90, p=2: Calibration Acc=0.9000, Brier=0.0741
Training Weighted KNN 130/150: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0756
Training Weighted KNN 131/150: k=11, σ=3.10, p=2
  K=11, σ=3.10, p=2: Calibration Acc=0.9000, Brier=0.0771
Training Weighted KNN 132/150: k=11, σ=3.20, p=2
  K=11, σ=3.20, p=2: Calibration Acc=0.9000, Brier=0.0786
Training Weighted KNN 133/150: k=11, σ=3.30, p=2
  K=11, σ=3.30, p=2: Calibration Acc=0.9000, Brier=0.0800
Training Weighted KNN 134/150: k=11, σ=3.40, p=2
  K=11, σ=3.40, p=2: Calibration Acc=0.9000, Brier=0.0815
Training Weighted KNN 135/150: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.0829
Training Weighted KNN 136/150: k=11, σ=3.60, p=2
  K=11, σ=3.60, p=2: Calibration Acc=0.9000, Brier=0.0843
Training Weighted KNN 137/150: k=11, σ=3.70, p=2
  K=11, σ=3.70, p=2: Calibration Acc=0.8500, Brier=0.0857
Training Weighted KNN 138/150: k=11, σ=3.80, p=2
  K=11, σ=3.80, p=2: Calibration Acc=0.8500, Brier=0.0870
Training Weighted KNN 139/150: k=11, σ=3.90, p=2
  K=11, σ=3.90, p=2: Calibration Acc=0.8500, Brier=0.0883
Training Weighted KNN 140/150: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=0.8500, Brier=0.0896
Training Weighted KNN 141/150: k=11, σ=4.10, p=2
  K=11, σ=4.10, p=2: Calibration Acc=0.9000, Brier=0.0907
Training Weighted KNN 142/150: k=11, σ=4.20, p=2
  K=11, σ=4.20, p=2: Calibration Acc=0.9000, Brier=0.0919
Training Weighted KNN 143/150: k=11, σ=4.30, p=2
  K=11, σ=4.30, p=2: Calibration Acc=0.9000, Brier=0.0930
Training Weighted KNN 144/150: k=11, σ=4.40, p=2
  K=11, σ=4.40, p=2: Calibration Acc=0.9000, Brier=0.0940
Training Weighted KNN 145/150: k=11, σ=4.50, p=2
  K=11, σ=4.50, p=2: Calibration Acc=0.9000, Brier=0.0951
Training Weighted KNN 146/150: k=11, σ=4.60, p=2
  K=11, σ=4.60, p=2: Calibration Acc=0.9000, Brier=0.0960
Training Weighted KNN 147/150: k=11, σ=4.70, p=2
  K=11, σ=4.70, p=2: Calibration Acc=0.9000, Brier=0.0969
Training Weighted KNN 148/150: k=11, σ=4.80, p=2
  K=11, σ=4.80, p=2: Calibration Acc=0.9000, Brier=0.0978
Training Weighted KNN 149/150: k=11, σ=4.90, p=2
  K=11, σ=4.90, p=2: Calibration Acc=0.9000, Brier=0.0987
Training Weighted KNN 150/150: k=11, σ=5.00, p=2
  K=11, σ=5.00, p=2: Calibration Acc=0.9000, Brier=0.0995

=== Training Complete ===
Process started with PID: 10666
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 12, 1: 8}
Minimum samples per class: 8

Class 1 calibration scores (8 points):
  Min: 0.2218
  Max: 0.9798
  Mean: 0.8443
  Std: 0.2384
  Percentiles: 25%=0.8739, 50%=0.9408, 75%=0.9614, 95%=0.9756, 99%=0.9790

Class 0 calibration scores (12 points):
  Min: 0.0000
  Max: 0.4755
  Mean: 0.1416
  Std: 0.1646

Class 1 probability distribution (dot plot):
0.0==================================================1.0
           ●
                                           ●
                                           ●
                                              ●
                                               ●
                                               ●
                                                ●
                                                ●
0.0==================================================1.0

Class 1 scores (sorted): [0.22178148422027624, 0.8685479438865805, 0.8756178423191286, 0.9382282503973255, 0.9434643516703501, 0.959277901563139, 0.9679294716386048, 0.9797821430568527]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0010278122699935992, 0.024723107138657118, 0.0257082274188348, 0.03914343789049971, 0.22225092655930617, 0.2335515059696544, 0.2974561212524024, 0.37981150336566905, 0.47551662543896683]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.5244833745610332, 1: 0.22178148422027624}
  Class 1 threshold: 0.22178148422027624
  α=0.01: threshold=0.2218
  α=0.01: storing threshold=0.2218
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.5244833745610332, 1: 0.22178148422027624}
  Class 1 threshold: 0.22178148422027624
  α=0.05: threshold=0.2218
  α=0.05: storing threshold=0.2218
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.620188496634331, 1: 0.22178148422027624}
  Class 1 threshold: 0.22178148422027624
  α=0.10: threshold=0.2218
  α=0.10: storing threshold=0.2218
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.7025438787475976, 1: 0.8685479438865805}
  Class 1 threshold: 0.8685479438865805
  α=0.20: threshold=0.8685
  α=0.20: storing threshold=0.8685
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.7025438787475976, 1: 0.8685479438865805}
  Class 1 threshold: 0.8685479438865805
  α=0.25: threshold=0.8685
  α=0.25: storing threshold=0.8685
HCD table has 332 rows
Using spatial coordinates for all 332 HCD points
Computing probabilities for 332 HCD points...
Probability matrix shape: (332, 2)
Export: Calibration class distribution: {0: 12, 1: 8}

Export: Class 1 calibration scores (8 points):
  Min: 0.2218
  Max: 0.9798
  Mean: 0.8443
  Std: 0.2384
  Percentiles: 25%=0.8739, 50%=0.9408, 75%=0.9614, 95%=0.9756, 99%=0.9790

Export: Class 0 calibration scores (12 points):
  Min: 0.0000
  Max: 0.4755
  Mean: 0.1416
  Std: 0.1646

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
           ●
                                           ●
                                           ●
                                              ●
                                               ●
                                               ●
                                                ●
                                                ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.22178148422027624, 0.8685479438865805, 0.8756178423191286, 0.9382282503973255, 0.9434643516703501, 0.959277901563139, 0.9679294716386048, 0.9797821430568527]

Export: Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0010278122699935992, 0.024723107138657118, 0.0257082274188348, 0.03914343789049971, 0.22225092655930617, 0.2335515059696544, 0.2974561212524024, 0.37981150336566905, 0.47551662543896683]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.22178148422027624
Export: α=0.01: threshold=0.2218
Export: α=0.01: storing threshold=0.2218
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.22178148422027624
Export: α=0.05: threshold=0.2218
Export: α=0.05: storing threshold=0.2218
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.22178148422027624
Export: α=0.10: threshold=0.2218
Export: α=0.10: storing threshold=0.2218
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.8685479438865805
Export: α=0.20: threshold=0.8685
Export: α=0.20: storing threshold=0.8685
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.8685479438865805
Export: α=0.25: threshold=0.8685
Export: α=0.25: storing threshold=0.8685

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      4 non-null      object 
 1   threshold_0.01  4 non-null      float64
 2   threshold_0.05  4 non-null      float64
 3   threshold_0.10  4 non-null      float64
 4   threshold_0.20  4 non-null      float64
 5   threshold_0.25  4 non-null      float64
dtypes: float64(5), object(1)
memory usage: 324.0+ bytes
None

Existing table content:
           model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0        knn_model_m1        0.566558  ...        0.569419        0.569419
1    knn_model_lipid3        0.351231  ...        0.648769        0.792679
2  knn_model_mystery1        0.000000  ...        0.000000        0.000000
3  knn_model_mystery2        0.000000  ...        0.000000        0.035919

[4 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
            model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_minerals1        0.221781  ...        0.868548        0.868548

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      5 non-null      object 
 1   threshold_0.01  5 non-null      float64
 2   threshold_0.05  5 non-null      float64
 3   threshold_0.10  5 non-null      float64
 4   threshold_0.20  5 non-null      float64
 5   threshold_0.25  5 non-null      float64
dtypes: float64(5), object(1)
memory usage: 372.0+ bytes
None

Merged table content:
            model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0         knn_model_m1        0.566558  ...        0.569419        0.569419
1     knn_model_lipid3        0.351231  ...        0.648769        0.792679
2   knn_model_mystery1        0.000000  ...        0.000000        0.000000
3   knn_model_mystery2        0.000000  ...        0.000000        0.035919
4  knn_model_minerals1        0.221781  ...        0.868548        0.868548

[5 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
TEST CALLBACK TRIGGERED: n_clicks=None
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
TEST CALLBACK TRIGGERED: n_clicks=None
measured_levels shape: (208,)
measured_class_labels shape: (126,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 125
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 126 but corresponding boolean dimension is 208
measured_levels shape: (208,)
measured_class_labels shape: (208,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 207
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Level 1: level_mask shape: (208,), sum: 28
Level 2: level_mask shape: (208,), sum: 51
Level 3: level_mask shape: (208,), sum: 70
Level 4: level_mask shape: (208,), sum: 30
measured_levels shape: (208,)
measured_class_labels shape: (208,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 207
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Level 1: level_mask shape: (208,), sum: 28
Level 2: level_mask shape: (208,), sum: 51
Level 3: level_mask shape: (208,), sum: 70
Level 4: level_mask shape: (208,), sum: 30
measured_levels shape: (208,)
measured_class_labels shape: (208,)
measured_indices shape: (208,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 207
measured_indices range: 0 to 311
Level 0: level_mask shape: (208,), sum: 29
Level 1: level_mask shape: (208,), sum: 28
Level 2: level_mask shape: (208,), sum: 51
Level 3: level_mask shape: (208,), sum: 70
Level 4: level_mask shape: (208,), sum: 30
measured_levels shape: (178,)
measured_class_labels shape: (178,)
measured_indices shape: (178,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 177
measured_indices range: 0 to 701
Level 0: level_mask shape: (178,), sum: 95
Level 1: level_mask shape: (178,), sum: 63
Level 2: level_mask shape: (178,), sum: 20

Starting Weighted KNN training process... using training dataset (158 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 158
Class distribution in training data: {0: 13, 1: 145}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 4
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 13, 1: 145}
Calibration set class distribution: {1: 20}
Training Weighted KNN 1/24: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 2/24: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 3/24: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 4/24: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 5/24: k=7, σ=2.50, p=1
  K=7, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 6/24: k=7, σ=3.00, p=1
  K=7, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 7/24: k=7, σ=3.50, p=1
  K=7, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 8/24: k=7, σ=4.00, p=1
  K=7, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 9/24: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0006
Training Weighted KNN 10/24: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 11/24: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 12/24: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 13/24: k=9, σ=2.50, p=1
  K=9, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 14/24: k=9, σ=3.00, p=1
  K=9, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 15/24: k=9, σ=3.50, p=1
  K=9, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 16/24: k=9, σ=4.00, p=1
  K=9, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 17/24: k=11, σ=0.50, p=1
  K=11, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 18/24: k=11, σ=1.00, p=1
  K=11, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 19/24: k=11, σ=1.50, p=1
  K=11, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 20/24: k=11, σ=2.00, p=1
  K=11, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 21/24: k=11, σ=2.50, p=1
  K=11, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 22/24: k=11, σ=3.00, p=1
  K=11, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 23/24: k=11, σ=3.50, p=1
  K=11, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 24/24: k=11, σ=4.00, p=1
  K=11, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0021

=== Training Complete ===
Process started with PID: 29790
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {1: 20}
Minimum samples per class: 20

Class 1 calibration scores (20 points):
  Min: 0.9211
  Max: 1.0000
  Mean: 0.9908
  Std: 0.0226
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 742 rows
Using spatial coordinates for all 742 HCD points
Computing probabilities for 742 HCD points...
Probability matrix shape: (742, 2)
Export: Calibration class distribution: {1: 20}

Export: Class 1 calibration scores (20 points):
  Min: 0.9211
  Max: 1.0000
  Mean: 0.9908
  Std: 0.0226
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity

Starting Weighted KNN training process... using training dataset (158 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 158
Class distribution in training data: {0: 13, 1: 145}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 4
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 13, 1: 145}
Calibration set class distribution: {1: 20}
Training Weighted KNN 1/24: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 2/24: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 3/24: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0011
Training Weighted KNN 4/24: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 5/24: k=7, σ=2.50, p=1
  K=7, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0016
Training Weighted KNN 6/24: k=7, σ=3.00, p=1
  K=7, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 7/24: k=7, σ=3.50, p=1
  K=7, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0019
Training Weighted KNN 8/24: k=7, σ=4.00, p=1
  K=7, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0021
Training Weighted KNN 9/24: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0006
Training Weighted KNN 10/24: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0007
Training Weighted KNN 11/24: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 12/24: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 13/24: k=9, σ=2.50, p=1
  K=9, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 14/24: k=9, σ=3.00, p=1
  K=9, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0013
Training Weighted KNN 15/24: k=9, σ=3.50, p=1
  K=9, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0014
Training Weighted KNN 16/24: k=9, σ=4.00, p=1
  K=9, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 17/24: k=11, σ=0.50, p=1
  K=11, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0009
Training Weighted KNN 18/24: k=11, σ=1.00, p=1
  K=11, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0010
Training Weighted KNN 19/24: k=11, σ=1.50, p=1
  K=11, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0012
Training Weighted KNN 20/24: k=11, σ=2.00, p=1
  K=11, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0015
Training Weighted KNN 21/24: k=11, σ=2.50, p=1
  K=11, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0017
Training Weighted KNN 22/24: k=11, σ=3.00, p=1
  K=11, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0018
Training Weighted KNN 23/24: k=11, σ=3.50, p=1
  K=11, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0020
Training Weighted KNN 24/24: k=11, σ=4.00, p=1
  K=11, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0021

=== Training Complete ===
Process started with PID: 29961
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {1: 20}
Minimum samples per class: 20

Class 1 calibration scores (20 points):
  Min: 0.9158
  Max: 1.0000
  Mean: 0.9900
  Std: 0.0247
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (0 points):
Error computing conformal thresholds: zero-size array to reduction operation minimum which has no identity
HCD table has 742 rows
Using spatial coordinates for all 742 HCD points
Computing probabilities for 742 HCD points...
Probability matrix shape: (742, 2)
Export: Calibration class distribution: {1: 20}

Export: Class 1 calibration scores (20 points):
  Min: 0.9158
  Max: 1.0000
  Mean: 0.9900
  Std: 0.0247
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (0 points):
Error in handle_model_export: zero-size array to reduction operation minimum which has no identity

Starting Weighted KNN training process... using training dataset (138 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 138
Class distribution in training data: {0: 12, 1: 126}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 4
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 12, 1: 126}
Calibration set class distribution: {0: 1, 1: 19}
Training Weighted KNN 1/24: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0136
Training Weighted KNN 2/24: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0143
Training Weighted KNN 3/24: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0153
Training Weighted KNN 4/24: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0164
Training Weighted KNN 5/24: k=7, σ=2.50, p=1
  K=7, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0175
Training Weighted KNN 6/24: k=7, σ=3.00, p=1
  K=7, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0185
Training Weighted KNN 7/24: k=7, σ=3.50, p=1
  K=7, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0194
Training Weighted KNN 8/24: k=7, σ=4.00, p=1
  K=7, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0202
Training Weighted KNN 9/24: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0132
Training Weighted KNN 10/24: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0138
Training Weighted KNN 11/24: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0147
Training Weighted KNN 12/24: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0157
Training Weighted KNN 13/24: k=9, σ=2.50, p=1
  K=9, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0166
Training Weighted KNN 14/24: k=9, σ=3.00, p=1
  K=9, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0175
Training Weighted KNN 15/24: k=9, σ=3.50, p=1
  K=9, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0182
Training Weighted KNN 16/24: k=9, σ=4.00, p=1
  K=9, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0189
Training Weighted KNN 17/24: k=11, σ=0.50, p=1
  K=11, σ=0.50, p=1: Calibration Acc=1.0000, Brier=0.0134
Training Weighted KNN 18/24: k=11, σ=1.00, p=1
  K=11, σ=1.00, p=1: Calibration Acc=1.0000, Brier=0.0141
Training Weighted KNN 19/24: k=11, σ=1.50, p=1
  K=11, σ=1.50, p=1: Calibration Acc=1.0000, Brier=0.0149
Training Weighted KNN 20/24: k=11, σ=2.00, p=1
  K=11, σ=2.00, p=1: Calibration Acc=1.0000, Brier=0.0158
Training Weighted KNN 21/24: k=11, σ=2.50, p=1
  K=11, σ=2.50, p=1: Calibration Acc=1.0000, Brier=0.0167
Training Weighted KNN 22/24: k=11, σ=3.00, p=1
  K=11, σ=3.00, p=1: Calibration Acc=1.0000, Brier=0.0175
Training Weighted KNN 23/24: k=11, σ=3.50, p=1
  K=11, σ=3.50, p=1: Calibration Acc=1.0000, Brier=0.0183
Training Weighted KNN 24/24: k=11, σ=4.00, p=1
  K=11, σ=4.00, p=1: Calibration Acc=1.0000, Brier=0.0189

=== Training Complete ===
Process started with PID: 30083
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 1, 1: 19}
Minimum samples per class: 1
Warning: Very few samples per class in holdout set

Class 1 calibration scores (19 points):
  Min: 0.6751
  Max: 1.0000
  Mean: 0.9560
  Std: 0.0975
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (1 points):
  Min: 0.2159
  Max: 0.2159
  Mean: 0.2159
  Std: 0.0000

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                                 ●
                                      ●
                                      ●
                                               ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.6751025757412507, 0.7650278677669047, 0.7661215202538081, 0.957986771448822, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.21593964767085527]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.7840603523291447, 1: 0.6751025757412507}
  Class 1 threshold: 0.6751025757412507
  α=0.01: threshold=0.6751
  α=0.01: storing threshold=0.6751
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.7840603523291447, 1: 0.6751025757412507}
  Class 1 threshold: 0.6751025757412507
  α=0.05: threshold=0.6751
  α=0.05: storing threshold=0.6751
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.7840603523291447, 1: 0.7650278677669047}
  Class 1 threshold: 0.7650278677669047
  α=0.10: threshold=0.7650
  α=0.10: storing threshold=0.7650
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.7840603523291447, 1: 0.957986771448822}
  Class 1 threshold: 0.957986771448822
  α=0.20: threshold=0.9580
  α=0.20: storing threshold=0.9580
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.7840603523291447, 1: 1.0}
  Class 1 threshold: 1.0
  α=0.25: threshold=1.0000
  α=0.25: storing threshold=1.0000
HCD table has 742 rows
Using spatial coordinates for all 742 HCD points
Computing probabilities for 742 HCD points...
Probability matrix shape: (742, 2)
Export: Calibration class distribution: {0: 1, 1: 19}

Export: Class 1 calibration scores (19 points):
  Min: 0.6751
  Max: 1.0000
  Mean: 0.9560
  Std: 0.0975
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (1 points):
  Min: 0.2159
  Max: 0.2159
  Mean: 0.2159
  Std: 0.0000

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
                                 ●
                                      ●
                                      ●
                                               ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.6751025757412507, 0.7650278677669047, 0.7661215202538081, 0.957986771448822, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.21593964767085527]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.6751025757412507
Export: α=0.01: threshold=0.6751
Export: α=0.01: storing threshold=0.6751
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.6751025757412507
Export: α=0.05: threshold=0.6751
Export: α=0.05: storing threshold=0.6751
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.7650278677669047
Export: α=0.10: threshold=0.7650
Export: α=0.10: storing threshold=0.7650
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.957986771448822
Export: α=0.20: threshold=0.9580
Export: α=0.20: storing threshold=0.9580
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 1.0
Export: α=0.25: threshold=1.0000
Export: α=0.25: storing threshold=1.0000
No 'class' column found, using 'selection' column instead
Error in update_level_breakdown: 'class'
No 'class' column found, using 'selection' column instead
Error in update_level_breakdown: 'class'
measured_levels shape: (356,)
measured_class_labels shape: (336,)
measured_indices shape: (356,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 335
measured_indices range: 0 to 731
Level 0: level_mask shape: (356,), sum: 95
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 336 but corresponding boolean dimension is 356
measured_levels shape: (356,)
measured_class_labels shape: (336,)
measured_indices shape: (356,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 335
measured_indices range: 0 to 731
Level 0: level_mask shape: (356,), sum: 95
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 336 but corresponding boolean dimension is 356
TEST CALLBACK TRIGGERED: n_clicks=1
TEST CALLBACK TRIGGERED: n_clicks=None
measured_levels shape: (356,)
measured_class_labels shape: (336,)
measured_indices shape: (356,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 335
measured_indices range: 0 to 731
Level 0: level_mask shape: (356,), sum: 95
Error in update_level_breakdown: boolean index did not match indexed array along dimension 0; dimension is 336 but corresponding boolean dimension is 356
measured_levels shape: (44,)
measured_class_labels shape: (44,)
measured_indices shape: (44,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 43
measured_indices range: 0 to 590
Level 0: level_mask shape: (44,), sum: 30
Level 1: level_mask shape: (44,), sum: 14

Starting Weighted KNN training process... using training dataset (30 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 30
Class distribution in training data: {0: 15, 1: 15}
K Min: 3
K Max: 10
K Step: 1
σ Min: 0.5
σ Max: 2
σ Step: 0.5
p: 1
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 14 points
Training set class distribution: {0: 15, 1: 15}
Calibration set class distribution: {0: 8, 1: 6}
Training Weighted KNN 1/32: k=3, σ=0.50, p=1
  K=3, σ=0.50, p=1: Calibration Acc=0.8571, Brier=0.0879
Training Weighted KNN 2/32: k=3, σ=1.00, p=1
  K=3, σ=1.00, p=1: Calibration Acc=0.8571, Brier=0.0880
Training Weighted KNN 3/32: k=3, σ=1.50, p=1
  K=3, σ=1.50, p=1: Calibration Acc=0.8571, Brier=0.0882
Training Weighted KNN 4/32: k=3, σ=2.00, p=1
  K=3, σ=2.00, p=1: Calibration Acc=0.8571, Brier=0.0885
Training Weighted KNN 5/32: k=4, σ=0.50, p=1
  K=4, σ=0.50, p=1: Calibration Acc=0.8571, Brier=0.1065
Training Weighted KNN 6/32: k=4, σ=1.00, p=1
  K=4, σ=1.00, p=1: Calibration Acc=0.8571, Brier=0.1069
Training Weighted KNN 7/32: k=4, σ=1.50, p=1
  K=4, σ=1.50, p=1: Calibration Acc=0.8571, Brier=0.1076
Training Weighted KNN 8/32: k=4, σ=2.00, p=1
  K=4, σ=2.00, p=1: Calibration Acc=0.8571, Brier=0.1086
Training Weighted KNN 9/32: k=5, σ=0.50, p=1
  K=5, σ=0.50, p=1: Calibration Acc=0.8571, Brier=0.1257
Training Weighted KNN 10/32: k=5, σ=1.00, p=1
  K=5, σ=1.00, p=1: Calibration Acc=0.8571, Brier=0.1265
Training Weighted KNN 11/32: k=5, σ=1.50, p=1
  K=5, σ=1.50, p=1: Calibration Acc=0.8571, Brier=0.1279
Training Weighted KNN 12/32: k=5, σ=2.00, p=1
  K=5, σ=2.00, p=1: Calibration Acc=0.8571, Brier=0.1298
Training Weighted KNN 13/32: k=6, σ=0.50, p=1
  K=6, σ=0.50, p=1: Calibration Acc=0.8571, Brier=0.1285
Training Weighted KNN 14/32: k=6, σ=1.00, p=1
  K=6, σ=1.00, p=1: Calibration Acc=0.8571, Brier=0.1293
Training Weighted KNN 15/32: k=6, σ=1.50, p=1
  K=6, σ=1.50, p=1: Calibration Acc=0.8571, Brier=0.1307
Training Weighted KNN 16/32: k=6, σ=2.00, p=1
  K=6, σ=2.00, p=1: Calibration Acc=0.8571, Brier=0.1325
Training Weighted KNN 17/32: k=7, σ=0.50, p=1
  K=7, σ=0.50, p=1: Calibration Acc=0.7857, Brier=0.1359
Training Weighted KNN 18/32: k=7, σ=1.00, p=1
  K=7, σ=1.00, p=1: Calibration Acc=0.7857, Brier=0.1369
Training Weighted KNN 19/32: k=7, σ=1.50, p=1
  K=7, σ=1.50, p=1: Calibration Acc=0.7857, Brier=0.1385
Training Weighted KNN 20/32: k=7, σ=2.00, p=1
  K=7, σ=2.00, p=1: Calibration Acc=0.7857, Brier=0.1406
Training Weighted KNN 21/32: k=8, σ=0.50, p=1
  K=8, σ=0.50, p=1: Calibration Acc=0.7857, Brier=0.1376
Training Weighted KNN 22/32: k=8, σ=1.00, p=1
  K=8, σ=1.00, p=1: Calibration Acc=0.7857, Brier=0.1386
Training Weighted KNN 23/32: k=8, σ=1.50, p=1
  K=8, σ=1.50, p=1: Calibration Acc=0.7857, Brier=0.1402
Training Weighted KNN 24/32: k=8, σ=2.00, p=1
  K=8, σ=2.00, p=1: Calibration Acc=0.7857, Brier=0.1423
Training Weighted KNN 25/32: k=9, σ=0.50, p=1
  K=9, σ=0.50, p=1: Calibration Acc=0.7857, Brier=0.1398
Training Weighted KNN 26/32: k=9, σ=1.00, p=1
  K=9, σ=1.00, p=1: Calibration Acc=0.7857, Brier=0.1408
Training Weighted KNN 27/32: k=9, σ=1.50, p=1
  K=9, σ=1.50, p=1: Calibration Acc=0.7857, Brier=0.1425
Training Weighted KNN 28/32: k=9, σ=2.00, p=1
  K=9, σ=2.00, p=1: Calibration Acc=0.7857, Brier=0.1447
Training Weighted KNN 29/32: k=10, σ=0.50, p=1
  K=10, σ=0.50, p=1: Calibration Acc=0.7857, Brier=0.1376
Training Weighted KNN 30/32: k=10, σ=1.00, p=1
  K=10, σ=1.00, p=1: Calibration Acc=0.7857, Brier=0.1386
Training Weighted KNN 31/32: k=10, σ=1.50, p=1
  K=10, σ=1.50, p=1: Calibration Acc=0.7857, Brier=0.1402
Training Weighted KNN 32/32: k=10, σ=2.00, p=1
  K=10, σ=2.00, p=1: Calibration Acc=0.7857, Brier=0.1422

=== Training Complete ===
Process started with PID: 42058
Processing training results

Starting Weighted KNN training process... using training dataset (30 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 30
Class distribution in training data: {0: 15, 1: 15}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 6
σ Step: 0.5
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 14 points
Training set class distribution: {0: 15, 1: 15}
Calibration set class distribution: {0: 8, 1: 6}
Training Weighted KNN 1/36: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.8571, Brier=0.1028
Training Weighted KNN 2/36: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.8571, Brier=0.1028
Training Weighted KNN 3/36: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.8571, Brier=0.1029
Training Weighted KNN 4/36: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.8571, Brier=0.1030
Training Weighted KNN 5/36: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.8571, Brier=0.1034
Training Weighted KNN 6/36: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.8571, Brier=0.1043
Training Weighted KNN 7/36: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=0.8571, Brier=0.1058
Training Weighted KNN 8/36: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=0.8571, Brier=0.1082
Training Weighted KNN 9/36: k=7, σ=4.50, p=2
  K=7, σ=4.50, p=2: Calibration Acc=0.8571, Brier=0.1118
Training Weighted KNN 10/36: k=7, σ=5.00, p=2
  K=7, σ=5.00, p=2: Calibration Acc=0.8571, Brier=0.1165
Training Weighted KNN 11/36: k=7, σ=5.50, p=2
  K=7, σ=5.50, p=2: Calibration Acc=0.8571, Brier=0.1221
Training Weighted KNN 12/36: k=7, σ=6.00, p=2
  K=7, σ=6.00, p=2: Calibration Acc=0.7857, Brier=0.1285
Training Weighted KNN 13/36: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.8571, Brier=0.1034
Training Weighted KNN 14/36: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.8571, Brier=0.1034
Training Weighted KNN 15/36: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.8571, Brier=0.1035
Training Weighted KNN 16/36: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.8571, Brier=0.1037
Training Weighted KNN 17/36: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.8571, Brier=0.1041
Training Weighted KNN 18/36: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=0.8571, Brier=0.1050
Training Weighted KNN 19/36: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=0.8571, Brier=0.1066
Training Weighted KNN 20/36: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=0.8571, Brier=0.1091
Training Weighted KNN 21/36: k=9, σ=4.50, p=2
  K=9, σ=4.50, p=2: Calibration Acc=0.8571, Brier=0.1128
Training Weighted KNN 22/36: k=9, σ=5.00, p=2
  K=9, σ=5.00, p=2: Calibration Acc=0.8571, Brier=0.1176
Training Weighted KNN 23/36: k=9, σ=5.50, p=2
  K=9, σ=5.50, p=2: Calibration Acc=0.7857, Brier=0.1234
Training Weighted KNN 24/36: k=9, σ=6.00, p=2
  K=9, σ=6.00, p=2: Calibration Acc=0.7857, Brier=0.1299
Training Weighted KNN 25/36: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.8571, Brier=0.1037
Training Weighted KNN 26/36: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.8571, Brier=0.1037
Training Weighted KNN 27/36: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.8571, Brier=0.1037
Training Weighted KNN 28/36: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.8571, Brier=0.1039
Training Weighted KNN 29/36: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.8571, Brier=0.1044
Training Weighted KNN 30/36: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=0.8571, Brier=0.1053
Training Weighted KNN 31/36: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=0.8571, Brier=0.1069
Training Weighted KNN 32/36: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=0.8571, Brier=0.1095
Training Weighted KNN 33/36: k=11, σ=4.50, p=2
  K=11, σ=4.50, p=2: Calibration Acc=0.8571, Brier=0.1132
Training Weighted KNN 34/36: k=11, σ=5.00, p=2
  K=11, σ=5.00, p=2: Calibration Acc=0.8571, Brier=0.1181
Training Weighted KNN 35/36: k=11, σ=5.50, p=2
  K=11, σ=5.50, p=2: Calibration Acc=0.7857, Brier=0.1239
Training Weighted KNN 36/36: k=11, σ=6.00, p=2
  K=11, σ=6.00, p=2: Calibration Acc=0.7857, Brier=0.1304

=== Training Complete ===
Process started with PID: 42119
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (14, 2)
Calibration labels shape: (14,)
Calibration class distribution: {0: 8, 1: 6}
Minimum samples per class: 6

Class 1 calibration scores (6 points):
  Min: 0.7451
  Max: 0.9880
  Mean: 0.8801
  Std: 0.0882
  Percentiles: 25%=0.8225, 50%=0.8777, 75%=0.9604, 95%=0.9875, 99%=0.9879

Class 0 calibration scores (8 points):
  Min: 0.0085
  Max: 0.6973
  Mean: 0.3254
  Std: 0.2395

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                                     ●
                                        ●
                                           ●
                                            ●
                                                 ●
                                                 ●
0.0==================================================1.0

Class 1 scores (sorted): [0.7450865107342102, 0.8059355960730521, 0.8723492876817597, 0.8830815149457607, 0.9861136355254106, 0.9880184173316645]
Class 0 scores (sorted): [0.008543967129722229, 0.07869549776227458, 0.17218804989386752, 0.2216109334093692, 0.33722058747839767, 0.4216166590411744, 0.6657419268558515, 0.697298426314338]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.302701573685662, 1: 0.7450865107342102}
  Class 1 threshold: 0.7450865107342102
  α=0.01: threshold=0.7451
  α=0.01: storing threshold=0.7451
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.302701573685662, 1: 0.7450865107342102}
  Class 1 threshold: 0.7450865107342102
  α=0.05: threshold=0.7451
  α=0.05: storing threshold=0.7451
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.302701573685662, 1: 0.7450865107342102}
  Class 1 threshold: 0.7450865107342102
  α=0.10: threshold=0.7451
  α=0.10: storing threshold=0.7451
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.3342580731441484, 1: 0.8059355960730521}
  Class 1 threshold: 0.8059355960730521
  α=0.20: threshold=0.8059
  α=0.20: storing threshold=0.8059
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.3342580731441484, 1: 0.8059355960730521}
  Class 1 threshold: 0.8059355960730521
  α=0.25: threshold=0.8059
  α=0.25: storing threshold=0.8059
HCD table has 766 rows
Using spatial coordinates for all 766 HCD points
Computing probabilities for 766 HCD points...
Probability matrix shape: (766, 2)
Export: Calibration class distribution: {0: 8, 1: 6}

Export: Class 1 calibration scores (6 points):
  Min: 0.7451
  Max: 0.9880
  Mean: 0.8801
  Std: 0.0882
  Percentiles: 25%=0.8225, 50%=0.8777, 75%=0.9604, 95%=0.9875, 99%=0.9879

Export: Class 0 calibration scores (8 points):
  Min: 0.0085
  Max: 0.6973
  Mean: 0.3254
  Std: 0.2395

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
                                     ●
                                        ●
                                           ●
                                            ●
                                                 ●
                                                 ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.7450865107342102, 0.8059355960730521, 0.8723492876817597, 0.8830815149457607, 0.9861136355254106, 0.9880184173316645]

Export: Class 0 scores (sorted): [0.008543967129722229, 0.07869549776227458, 0.17218804989386752, 0.2216109334093692, 0.33722058747839767, 0.4216166590411744, 0.6657419268558515, 0.697298426314338]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.7450865107342102
Export: α=0.01: threshold=0.7451
Export: α=0.01: storing threshold=0.7451
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.7450865107342102
Export: α=0.05: threshold=0.7451
Export: α=0.05: storing threshold=0.7451
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.7450865107342102
Export: α=0.10: threshold=0.7451
Export: α=0.10: storing threshold=0.7451
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.8059355960730521
Export: α=0.20: threshold=0.8059
Export: α=0.20: storing threshold=0.8059
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.8059355960730521
Export: α=0.25: threshold=0.8059
Export: α=0.25: storing threshold=0.8059
measured_levels shape: (129,)
measured_class_labels shape: (129,)
measured_indices shape: (129,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 128
measured_indices range: 0 to 655
Level 0: level_mask shape: (129,), sum: 30
Level 1: level_mask shape: (129,), sum: 14
Level 2: level_mask shape: (129,), sum: 7
Level 3: level_mask shape: (129,), sum: 11
Level 4: level_mask shape: (129,), sum: 21
Level 5: level_mask shape: (129,), sum: 35
Level 6: level_mask shape: (129,), sum: 11

Starting Weighted KNN training process... using training dataset (118 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 118
Class distribution in training data: {0: 32, 1: 86}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 6
σ Step: 0.5
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 11 points
Training set class distribution: {0: 32, 1: 86}
Calibration set class distribution: {0: 4, 1: 7}
Training Weighted KNN 1/36: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0339
Training Weighted KNN 2/36: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0337
Training Weighted KNN 3/36: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0329
Training Weighted KNN 4/36: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0320
Training Weighted KNN 5/36: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0322
Training Weighted KNN 6/36: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0344
Training Weighted KNN 7/36: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0384
Training Weighted KNN 8/36: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0432
Training Weighted KNN 9/36: k=7, σ=4.50, p=2
  K=7, σ=4.50, p=2: Calibration Acc=1.0000, Brier=0.0483
Training Weighted KNN 10/36: k=7, σ=5.00, p=2
  K=7, σ=5.00, p=2: Calibration Acc=1.0000, Brier=0.0531
Training Weighted KNN 11/36: k=7, σ=5.50, p=2
  K=7, σ=5.50, p=2: Calibration Acc=0.9091, Brier=0.0575
Training Weighted KNN 12/36: k=7, σ=6.00, p=2
  K=7, σ=6.00, p=2: Calibration Acc=0.9091, Brier=0.0614
Training Weighted KNN 13/36: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0336
Training Weighted KNN 14/36: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0334
Training Weighted KNN 15/36: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0325
Training Weighted KNN 16/36: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0315
Training Weighted KNN 17/36: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0315
Training Weighted KNN 18/36: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0334
Training Weighted KNN 19/36: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0368
Training Weighted KNN 20/36: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0410
Training Weighted KNN 21/36: k=9, σ=4.50, p=2
  K=9, σ=4.50, p=2: Calibration Acc=1.0000, Brier=0.0453
Training Weighted KNN 22/36: k=9, σ=5.00, p=2
  K=9, σ=5.00, p=2: Calibration Acc=1.0000, Brier=0.0493
Training Weighted KNN 23/36: k=9, σ=5.50, p=2
  K=9, σ=5.50, p=2: Calibration Acc=0.9091, Brier=0.0529
Training Weighted KNN 24/36: k=9, σ=6.00, p=2
  K=9, σ=6.00, p=2: Calibration Acc=0.9091, Brier=0.0559
Training Weighted KNN 25/36: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=1.0000, Brier=0.0342
Training Weighted KNN 26/36: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=1.0000, Brier=0.0340
Training Weighted KNN 27/36: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=1.0000, Brier=0.0333
Training Weighted KNN 28/36: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=1.0000, Brier=0.0326
Training Weighted KNN 29/36: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=1.0000, Brier=0.0332
Training Weighted KNN 30/36: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=1.0000, Brier=0.0360
Training Weighted KNN 31/36: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=1.0000, Brier=0.0407
Training Weighted KNN 32/36: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=1.0000, Brier=0.0464
Training Weighted KNN 33/36: k=11, σ=4.50, p=2
  K=11, σ=4.50, p=2: Calibration Acc=1.0000, Brier=0.0524
Training Weighted KNN 34/36: k=11, σ=5.00, p=2
  K=11, σ=5.00, p=2: Calibration Acc=0.9091, Brier=0.0582
Training Weighted KNN 35/36: k=11, σ=5.50, p=2
  K=11, σ=5.50, p=2: Calibration Acc=0.9091, Brier=0.0636
Training Weighted KNN 36/36: k=11, σ=6.00, p=2
  K=11, σ=6.00, p=2: Calibration Acc=0.9091, Brier=0.0685

=== Training Complete ===
Process started with PID: 44382
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (11, 2)
Calibration labels shape: (11,)
Calibration class distribution: {0: 4, 1: 7}
Minimum samples per class: 4

Class 1 calibration scores (7 points):
  Min: 0.5421
  Max: 1.0000
  Mean: 0.9016
  Std: 0.1667
  Percentiles: 25%=0.8844, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (4 points):
  Min: 0.0000
  Max: 0.2901
  Mean: 0.1236
  Std: 0.1150

Class 1 probability distribution (dot plot):
0.0==================================================1.0
                           ●
                                      ●
                                                 ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.5421304219250319, 0.7702850744495643, 0.9984752585232598, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.03520827146209431, 0.1691350480757996, 0.29010573846181265]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.7098942615381874, 1: 0.5421304219250319}
  Class 1 threshold: 0.5421304219250319
  α=0.01: threshold=0.5421
  α=0.01: storing threshold=0.5421
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.7098942615381874, 1: 0.5421304219250319}
  Class 1 threshold: 0.5421304219250319
  α=0.05: threshold=0.5421
  α=0.05: storing threshold=0.5421
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.7098942615381874, 1: 0.5421304219250319}
  Class 1 threshold: 0.5421304219250319
  α=0.10: threshold=0.5421
  α=0.10: storing threshold=0.5421
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.7098942615381874, 1: 0.7702850744495643}
  Class 1 threshold: 0.7702850744495643
  α=0.20: threshold=0.7703
  α=0.20: storing threshold=0.7703
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.7098942615381874, 1: 0.7702850744495643}
  Class 1 threshold: 0.7702850744495643
  α=0.25: threshold=0.7703
  α=0.25: storing threshold=0.7703
HCD table has 766 rows
Using spatial coordinates for all 766 HCD points
Computing probabilities for 766 HCD points...
Probability matrix shape: (766, 2)
Export: Calibration class distribution: {0: 4, 1: 7}

Export: Class 1 calibration scores (7 points):
  Min: 0.5421
  Max: 1.0000
  Mean: 0.9016
  Std: 0.1667
  Percentiles: 25%=0.8844, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (4 points):
  Min: 0.0000
  Max: 0.2901
  Mean: 0.1236
  Std: 0.1150

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
                           ●
                                      ●
                                                 ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.5421304219250319, 0.7702850744495643, 0.9984752585232598, 1.0, 1.0, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.0, 0.03520827146209431, 0.1691350480757996, 0.29010573846181265]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.5421304219250319
Export: α=0.01: threshold=0.5421
Export: α=0.01: storing threshold=0.5421
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.5421304219250319
Export: α=0.05: threshold=0.5421
Export: α=0.05: storing threshold=0.5421
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.5421304219250319
Export: α=0.10: threshold=0.5421
Export: α=0.10: storing threshold=0.5421
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.7702850744495643
Export: α=0.20: threshold=0.7703
Export: α=0.20: storing threshold=0.7703
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.7702850744495643
Export: α=0.25: threshold=0.7703
Export: α=0.25: storing threshold=0.7703

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 180.0+ bytes
None

Existing table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m1        0.745087  ...        0.805936        0.805936

[1 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m2         0.54213  ...        0.770285        0.770285

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      2 non-null      object 
 1   threshold_0.01  2 non-null      float64
 2   threshold_0.05  2 non-null      float64
 3   threshold_0.10  2 non-null      float64
 4   threshold_0.20  2 non-null      float64
 5   threshold_0.25  2 non-null      float64
dtypes: float64(5), object(1)
memory usage: 228.0+ bytes
None

Merged table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m1        0.745087  ...        0.805936        0.805936
1  knn_model_m2        0.542130  ...        0.770285        0.770285

[2 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'
measured_levels shape: (275,)
measured_class_labels shape: (275,)
measured_indices shape: (275,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 274
measured_indices range: 0 to 734
Level 0: level_mask shape: (275,), sum: 30
Level 1: level_mask shape: (275,), sum: 14
Level 2: level_mask shape: (275,), sum: 10
Level 3: level_mask shape: (275,), sum: 19
Level 4: level_mask shape: (275,), sum: 27
Level 5: level_mask shape: (275,), sum: 37
Level 6: level_mask shape: (275,), sum: 54
Level 7: level_mask shape: (275,), sum: 64
Level 8: level_mask shape: (275,), sum: 20

Starting Weighted KNN training process... using training dataset (255 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 255
Class distribution in training data: {0: 197, 1: 58}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 6
σ Step: 0.5
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 20 points
Training set class distribution: {0: 197, 1: 58}
Calibration set class distribution: {0: 14, 1: 6}
Training Weighted KNN 1/36: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.9000, Brier=0.0467
Training Weighted KNN 2/36: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0474
Training Weighted KNN 3/36: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0511
Training Weighted KNN 4/36: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0561
Training Weighted KNN 5/36: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0598
Training Weighted KNN 6/36: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0621
Training Weighted KNN 7/36: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.0633
Training Weighted KNN 8/36: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.0641
Training Weighted KNN 9/36: k=7, σ=4.50, p=2
  K=7, σ=4.50, p=2: Calibration Acc=0.9000, Brier=0.0645
Training Weighted KNN 10/36: k=7, σ=5.00, p=2
  K=7, σ=5.00, p=2: Calibration Acc=0.9000, Brier=0.0648
Training Weighted KNN 11/36: k=7, σ=5.50, p=2
  K=7, σ=5.50, p=2: Calibration Acc=0.9000, Brier=0.0649
Training Weighted KNN 12/36: k=7, σ=6.00, p=2
  K=7, σ=6.00, p=2: Calibration Acc=0.9000, Brier=0.0650
Training Weighted KNN 13/36: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.8500, Brier=0.0471
Training Weighted KNN 14/36: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0480
Training Weighted KNN 15/36: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0522
Training Weighted KNN 16/36: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0579
Training Weighted KNN 17/36: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0624
Training Weighted KNN 18/36: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0652
Training Weighted KNN 19/36: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.0669
Training Weighted KNN 20/36: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.0680
Training Weighted KNN 21/36: k=9, σ=4.50, p=2
  K=9, σ=4.50, p=2: Calibration Acc=0.9000, Brier=0.0686
Training Weighted KNN 22/36: k=9, σ=5.00, p=2
  K=9, σ=5.00, p=2: Calibration Acc=0.9000, Brier=0.0690
Training Weighted KNN 23/36: k=9, σ=5.50, p=2
  K=9, σ=5.50, p=2: Calibration Acc=0.9000, Brier=0.0692
Training Weighted KNN 24/36: k=9, σ=6.00, p=2
  K=9, σ=6.00, p=2: Calibration Acc=0.9000, Brier=0.0694
Training Weighted KNN 25/36: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.8500, Brier=0.0477
Training Weighted KNN 26/36: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.9000, Brier=0.0489
Training Weighted KNN 27/36: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.9000, Brier=0.0539
Training Weighted KNN 28/36: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.9000, Brier=0.0608
Training Weighted KNN 29/36: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.9000, Brier=0.0663
Training Weighted KNN 30/36: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=0.9000, Brier=0.0698
Training Weighted KNN 31/36: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=0.9000, Brier=0.0719
Training Weighted KNN 32/36: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=0.9000, Brier=0.0732
Training Weighted KNN 33/36: k=11, σ=4.50, p=2
  K=11, σ=4.50, p=2: Calibration Acc=0.9000, Brier=0.0739
Training Weighted KNN 34/36: k=11, σ=5.00, p=2
  K=11, σ=5.00, p=2: Calibration Acc=0.9000, Brier=0.0744
Training Weighted KNN 35/36: k=11, σ=5.50, p=2
  K=11, σ=5.50, p=2: Calibration Acc=0.9000, Brier=0.0746
Training Weighted KNN 36/36: k=11, σ=6.00, p=2
  K=11, σ=6.00, p=2: Calibration Acc=0.9000, Brier=0.0748

=== Training Complete ===
Process started with PID: 45436
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (20, 2)
Calibration labels shape: (20,)
Calibration class distribution: {0: 14, 1: 6}
Minimum samples per class: 6

Class 1 calibration scores (6 points):
  Min: 0.2960
  Max: 1.0000
  Mean: 0.8827
  Std: 0.2624
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (14 points):
  Min: 0.0000
  Max: 0.7037
  Mean: 0.1325
  Std: 0.1964

Class 1 probability distribution (dot plot):
0.0==================================================1.0
              ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.29595510392752045, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12474643062692636, 0.12753460762565538, 0.1486490396468733, 0.1565642936404712, 0.16379853317754686, 0.43051459621177407, 0.703721991965894]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.2962780080341059, 1: 0.29595510392752045}
  Class 1 threshold: 0.29595510392752045
  α=0.01: threshold=0.2960
  α=0.01: storing threshold=0.2960
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.2962780080341059, 1: 0.29595510392752045}
  Class 1 threshold: 0.29595510392752045
  α=0.05: threshold=0.2960
  α=0.05: storing threshold=0.2960
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.5694854037882259, 1: 0.29595510392752045}
  Class 1 threshold: 0.29595510392752045
  α=0.10: threshold=0.2960
  α=0.10: storing threshold=0.2960
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.8362014668224532, 1: 1.0}
  Class 1 threshold: 1.0
  α=0.20: threshold=1.0000
  α=0.20: storing threshold=1.0000
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.8434357063595288, 1: 1.0}
  Class 1 threshold: 1.0
  α=0.25: threshold=1.0000
  α=0.25: storing threshold=1.0000
HCD table has 766 rows
Using spatial coordinates for all 766 HCD points
Computing probabilities for 766 HCD points...
Probability matrix shape: (766, 2)
Export: Calibration class distribution: {0: 14, 1: 6}

Export: Class 1 calibration scores (6 points):
  Min: 0.2960
  Max: 1.0000
  Mean: 0.8827
  Std: 0.2624
  Percentiles: 25%=1.0000, 50%=1.0000, 75%=1.0000, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (14 points):
  Min: 0.0000
  Max: 0.7037
  Mean: 0.1325
  Std: 0.1964

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
              ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.29595510392752045, 1.0, 1.0, 1.0, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12474643062692636, 0.12753460762565538, 0.1486490396468733, 0.1565642936404712, 0.16379853317754686, 0.43051459621177407, 0.703721991965894]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.29595510392752045
Export: α=0.01: threshold=0.2960
Export: α=0.01: storing threshold=0.2960
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.29595510392752045
Export: α=0.05: threshold=0.2960
Export: α=0.05: storing threshold=0.2960
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.29595510392752045
Export: α=0.10: threshold=0.2960
Export: α=0.10: storing threshold=0.2960
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 1.0
Export: α=0.20: threshold=1.0000
Export: α=0.20: storing threshold=1.0000
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 1.0
Export: α=0.25: threshold=1.0000
Export: α=0.25: storing threshold=1.0000

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      2 non-null      object 
 1   threshold_0.01  2 non-null      float64
 2   threshold_0.05  2 non-null      float64
 3   threshold_0.10  2 non-null      float64
 4   threshold_0.20  2 non-null      float64
 5   threshold_0.25  2 non-null      float64
dtypes: float64(5), object(1)
memory usage: 228.0+ bytes
None

Existing table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m1        0.745087  ...        0.805936        0.805936
1  knn_model_m2        0.542130  ...        0.770285        0.770285

[2 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m3        0.295955  ...             1.0             1.0

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      3 non-null      object 
 1   threshold_0.01  3 non-null      float64
 2   threshold_0.05  3 non-null      float64
 3   threshold_0.10  3 non-null      float64
 4   threshold_0.20  3 non-null      float64
 5   threshold_0.25  3 non-null      float64
dtypes: float64(5), object(1)
memory usage: 276.0+ bytes
None

Merged table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m1        0.745087  ...        0.805936        0.805936
1  knn_model_m2        0.542130  ...        0.770285        0.770285
2  knn_model_m3        0.295955  ...        1.000000        1.000000

[3 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'
measured_levels shape: (435,)
measured_class_labels shape: (435,)
measured_indices shape: (435,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 434
measured_indices range: 0 to 680
Level 0: level_mask shape: (435,), sum: 29
Level 1: level_mask shape: (435,), sum: 13
Level 2: level_mask shape: (435,), sum: 22
Level 3: level_mask shape: (435,), sum: 33
Level 4: level_mask shape: (435,), sum: 38
Level 5: level_mask shape: (435,), sum: 69
Level 6: level_mask shape: (435,), sum: 89
Level 7: level_mask shape: (435,), sum: 142
measured_levels shape: (435,)
measured_class_labels shape: (435,)
measured_indices shape: (435,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 434
measured_indices range: 0 to 680
Level 0: level_mask shape: (435,), sum: 29
Level 1: level_mask shape: (435,), sum: 13
Level 2: level_mask shape: (435,), sum: 22
Level 3: level_mask shape: (435,), sum: 33
Level 4: level_mask shape: (435,), sum: 38
Level 5: level_mask shape: (435,), sum: 69
Level 6: level_mask shape: (435,), sum: 89
Level 7: level_mask shape: (435,), sum: 142

Starting Weighted KNN training process... using training dataset (293 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 293
Class distribution in training data: {0: 156, 1: 137}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 6
σ Step: 0.5
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 142 points
Training set class distribution: {0: 156, 1: 137}
Calibration set class distribution: {0: 67, 1: 75}
Training Weighted KNN 1/36: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.7042, Brier=0.1996
Training Weighted KNN 2/36: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.7113, Brier=0.1972
Training Weighted KNN 3/36: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.7254, Brier=0.1920
Training Weighted KNN 4/36: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.7183, Brier=0.1872
Training Weighted KNN 5/36: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.7183, Brier=0.1839
Training Weighted KNN 6/36: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.7183, Brier=0.1822
Training Weighted KNN 7/36: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=0.7113, Brier=0.1814
Training Weighted KNN 8/36: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=0.7183, Brier=0.1810
Training Weighted KNN 9/36: k=7, σ=4.50, p=2
  K=7, σ=4.50, p=2: Calibration Acc=0.7254, Brier=0.1808
Training Weighted KNN 10/36: k=7, σ=5.00, p=2
  K=7, σ=5.00, p=2: Calibration Acc=0.7254, Brier=0.1807
Training Weighted KNN 11/36: k=7, σ=5.50, p=2
  K=7, σ=5.50, p=2: Calibration Acc=0.7254, Brier=0.1807
Training Weighted KNN 12/36: k=7, σ=6.00, p=2
  K=7, σ=6.00, p=2: Calibration Acc=0.7254, Brier=0.1807
Training Weighted KNN 13/36: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.7042, Brier=0.1980
Training Weighted KNN 14/36: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.7042, Brier=0.1954
Training Weighted KNN 15/36: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.7324, Brier=0.1897
Training Weighted KNN 16/36: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.7254, Brier=0.1839
Training Weighted KNN 17/36: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.7324, Brier=0.1797
Training Weighted KNN 18/36: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=0.7324, Brier=0.1771
Training Weighted KNN 19/36: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=0.7324, Brier=0.1757
Training Weighted KNN 20/36: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=0.7254, Brier=0.1750
Training Weighted KNN 21/36: k=9, σ=4.50, p=2
  K=9, σ=4.50, p=2: Calibration Acc=0.7394, Brier=0.1746
Training Weighted KNN 22/36: k=9, σ=5.00, p=2
  K=9, σ=5.00, p=2: Calibration Acc=0.7394, Brier=0.1744
Training Weighted KNN 23/36: k=9, σ=5.50, p=2
  K=9, σ=5.50, p=2: Calibration Acc=0.7465, Brier=0.1742
Training Weighted KNN 24/36: k=9, σ=6.00, p=2
  K=9, σ=6.00, p=2: Calibration Acc=0.7465, Brier=0.1741
Training Weighted KNN 25/36: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.7042, Brier=0.1970
Training Weighted KNN 26/36: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.7042, Brier=0.1943
Training Weighted KNN 27/36: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.7254, Brier=0.1883
Training Weighted KNN 28/36: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.7254, Brier=0.1821
Training Weighted KNN 29/36: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.7183, Brier=0.1774
Training Weighted KNN 30/36: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=0.7183, Brier=0.1745
Training Weighted KNN 31/36: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=0.7183, Brier=0.1729
Training Weighted KNN 32/36: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=0.7324, Brier=0.1720
Training Weighted KNN 33/36: k=11, σ=4.50, p=2
  K=11, σ=4.50, p=2: Calibration Acc=0.7676, Brier=0.1714
Training Weighted KNN 34/36: k=11, σ=5.00, p=2
  K=11, σ=5.00, p=2: Calibration Acc=0.7676, Brier=0.1711
Training Weighted KNN 35/36: k=11, σ=5.50, p=2
  K=11, σ=5.50, p=2: Calibration Acc=0.7676, Brier=0.1709
Training Weighted KNN 36/36: k=11, σ=6.00, p=2
  K=11, σ=6.00, p=2: Calibration Acc=0.7676, Brier=0.1707

=== Training Complete ===
Process started with PID: 49007
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (142, 2)
Calibration labels shape: (142,)
Calibration class distribution: {0: 67, 1: 75}
Minimum samples per class: 67

Class 1 calibration scores (75 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.6258
  Std: 0.2845
  Percentiles: 25%=0.4299, 50%=0.6643, 75%=0.8844, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (67 points):
  Min: 0.0000
  Max: 0.7657
  Mean: 0.2962
  Std: 0.1850

Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
     ●
     ●
     ●
     ●
          ●
          ●
           ●
           ●
           ●
           ●
           ●
                ●
                ●
                ●
                ●
                 ●
                    ●
                     ●
                     ●
                     ●
                      ●
                      ●
                      ●
                      ●
                       ●
                       ●
                           ●
                           ●
                           ●
                           ●
                           ●
                           ●
                            ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                       ●
                                       ●
                                       ●
                                       ●
                                       ●
                                        ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.0, 0.10072907062273348, 0.11559994004330192, 0.11592585062931235, 0.11757332557207877, 0.20974369845925783, 0.21653337553727942, 0.2228867927547747, 0.22465608820957106, 0.2247535281752427, 0.23027993079179965, 0.23956899498883705, 0.3219835947355759, 0.32283584094044204, 0.3288952063671216, 0.338799563763066, 0.34050788061231346, 0.40367893701928764, 0.4285181246488427, 0.43138047817556624, 0.43833393082839106, 0.44040869749381867, 0.4441263079059028, 0.453035079772861, 0.4558925729183532, 0.4629078924086329, 0.4757077648350192, 0.5450397434520231, 0.5468209332019921, 0.5488817979740723, 0.5522955886329814, 0.5564272978884824, 0.5590038689103934, 0.562813896598092, 0.6604906082209684, 0.6631605561791587, 0.6633757159394215, 0.6642844838136124, 0.6664603063019384, 0.6762707682828074, 0.6764053673822141, 0.7692027718312571, 0.7721813620691736, 0.775682366517672, 0.7771858786040581, 0.777514755111747, 0.7799172532460125, 0.7808687381480711, 0.783038187065091, 0.7840077529201269, 0.7914549380645154, 0.7943667415203364, 0.8079051269117312, 0.8835981538326494, 0.883941507956778, 0.8843862692640966, 0.8844290922948096, 0.8902493219192977, 0.8919460272615783, 0.892634244572674, 0.8933968137202549, 0.8950838315978227, 0.8976560342619543, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10155492427370319, 0.10299218666139341, 0.1038149478853384, 0.11191587145737314, 0.1140148692463969, 0.11485370815467398, 0.11545065192263913, 0.11548797621731437, 0.1176704150369535, 0.11822029574423332, 0.118303094782885, 0.12814205211424223, 0.20304822008820386, 0.20686839775155955, 0.210193644030739, 0.2104395947767059, 0.21465220986342476, 0.21512686547482499, 0.21675696793267069, 0.21826610920703, 0.21896279191812779, 0.22038823878995753, 0.2210552695781831, 0.2231376431416877, 0.2235941030175338, 0.22998249137453367, 0.23192965497343837, 0.31350184644318835, 0.31520119216499615, 0.32269908407907866, 0.32412564399373966, 0.3295091735777989, 0.33046853115091074, 0.33177060207399384, 0.33433667652041876, 0.33604758578155297, 0.336552677654804, 0.3382717020161366, 0.34378028898129, 0.35171904107358437, 0.36366783387786344, 0.4312655124696057, 0.43126848874562823, 0.43390219717585077, 0.4391118238630242, 0.4414585857860794, 0.4443660870598525, 0.46352632981677316, 0.4689904314757097, 0.4727542728227333, 0.4729138923967342, 0.4904589510020313, 0.5469978511744416, 0.5511107060372887, 0.5578998528409116, 0.5669991320364844, 0.5672213303698644, 0.6632007074577599, 0.6655776554317187, 0.6693564403933197, 0.7657253521905155]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.23427464780948462, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.01: threshold=0.0000
  α=0.01: storing threshold=0.0000
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.33679929254224006, 1: 0.11592585062931235}
  Class 1 threshold: 0.11592585062931235
  α=0.05: threshold=0.1159
  α=0.05: storing threshold=0.1159
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.4421001471590884, 1: 0.2228867927547747}
  Class 1 threshold: 0.2228867927547747
  α=0.10: threshold=0.2229
  α=0.10: storing threshold=0.2229
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.5364736701832267, 1: 0.3288952063671216}
  Class 1 threshold: 0.3288952063671216
  α=0.20: threshold=0.3289
  α=0.20: storing threshold=0.3289
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.5608881761369757, 1: 0.4285181246488427}
  Class 1 threshold: 0.4285181246488427
  α=0.25: threshold=0.4285
  α=0.25: storing threshold=0.4285
HCD table has 752 rows
Using spatial coordinates for all 752 HCD points
Computing probabilities for 752 HCD points...
Probability matrix shape: (752, 2)
Export: Calibration class distribution: {0: 67, 1: 75}

Export: Class 1 calibration scores (75 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.6258
  Std: 0.2845
  Percentiles: 25%=0.4299, 50%=0.6643, 75%=0.8844, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (67 points):
  Min: 0.0000
  Max: 0.7657
  Mean: 0.2962
  Std: 0.1850

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
     ●
     ●
     ●
     ●
          ●
          ●
           ●
           ●
           ●
           ●
           ●
                ●
                ●
                ●
                ●
                 ●
                    ●
                     ●
                     ●
                     ●
                      ●
                      ●
                      ●
                      ●
                       ●
                       ●
                           ●
                           ●
                           ●
                           ●
                           ●
                           ●
                            ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                       ●
                                       ●
                                       ●
                                       ●
                                       ●
                                        ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.0, 0.10072907062273348, 0.11559994004330192, 0.11592585062931235, 0.11757332557207877, 0.20974369845925783, 0.21653337553727942, 0.2228867927547747, 0.22465608820957106, 0.2247535281752427, 0.23027993079179965, 0.23956899498883705, 0.3219835947355759, 0.32283584094044204, 0.3288952063671216, 0.338799563763066, 0.34050788061231346, 0.40367893701928764, 0.4285181246488427, 0.43138047817556624, 0.43833393082839106, 0.44040869749381867, 0.4441263079059028, 0.453035079772861, 0.4558925729183532, 0.4629078924086329, 0.4757077648350192, 0.5450397434520231, 0.5468209332019921, 0.5488817979740723, 0.5522955886329814, 0.5564272978884824, 0.5590038689103934, 0.562813896598092, 0.6604906082209684, 0.6631605561791587, 0.6633757159394215, 0.6642844838136124, 0.6664603063019384, 0.6762707682828074, 0.6764053673822141, 0.7692027718312571, 0.7721813620691736, 0.775682366517672, 0.7771858786040581, 0.777514755111747, 0.7799172532460125, 0.7808687381480711, 0.783038187065091, 0.7840077529201269, 0.7914549380645154, 0.7943667415203364, 0.8079051269117312, 0.8835981538326494, 0.883941507956778, 0.8843862692640966, 0.8844290922948096, 0.8902493219192977, 0.8919460272615783, 0.892634244572674, 0.8933968137202549, 0.8950838315978227, 0.8976560342619543, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10155492427370319, 0.10299218666139341, 0.1038149478853384, 0.11191587145737314, 0.1140148692463969, 0.11485370815467398, 0.11545065192263913, 0.11548797621731437, 0.1176704150369535, 0.11822029574423332, 0.118303094782885, 0.12814205211424223, 0.20304822008820386, 0.20686839775155955, 0.210193644030739, 0.2104395947767059, 0.21465220986342476, 0.21512686547482499, 0.21675696793267069, 0.21826610920703, 0.21896279191812779, 0.22038823878995753, 0.2210552695781831, 0.2231376431416877, 0.2235941030175338, 0.22998249137453367, 0.23192965497343837, 0.31350184644318835, 0.31520119216499615, 0.32269908407907866, 0.32412564399373966, 0.3295091735777989, 0.33046853115091074, 0.33177060207399384, 0.33433667652041876, 0.33604758578155297, 0.336552677654804, 0.3382717020161366, 0.34378028898129, 0.35171904107358437, 0.36366783387786344, 0.4312655124696057, 0.43126848874562823, 0.43390219717585077, 0.4391118238630242, 0.4414585857860794, 0.4443660870598525, 0.46352632981677316, 0.4689904314757097, 0.4727542728227333, 0.4729138923967342, 0.4904589510020313, 0.5469978511744416, 0.5511107060372887, 0.5578998528409116, 0.5669991320364844, 0.5672213303698644, 0.6632007074577599, 0.6655776554317187, 0.6693564403933197, 0.7657253521905155]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.0
Export: α=0.01: threshold=0.0000
Export: α=0.01: storing threshold=0.0000
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.11592585062931235
Export: α=0.05: threshold=0.1159
Export: α=0.05: storing threshold=0.1159
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.2228867927547747
Export: α=0.10: threshold=0.2229
Export: α=0.10: storing threshold=0.2229
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.3288952063671216
Export: α=0.20: threshold=0.3289
Export: α=0.20: storing threshold=0.3289
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.4285181246488427
Export: α=0.25: threshold=0.4285
Export: α=0.25: storing threshold=0.4285
Computing conformal thresholds for selected model...
Calibration data shape: (142, 2)
Calibration labels shape: (142,)
Calibration class distribution: {0: 67, 1: 75}
Minimum samples per class: 67

Class 1 calibration scores (75 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.6257
  Std: 0.2843
  Percentiles: 25%=0.4339, 50%=0.6649, 75%=0.8856, 95%=1.0000, 99%=1.0000

Class 0 calibration scores (67 points):
  Min: 0.0000
  Max: 0.7689
  Mean: 0.2960
  Std: 0.1847

Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
     ●
     ●
     ●
     ●
          ●
          ●
           ●
           ●
           ●
           ●
           ●
                ●
                ●
                ●
                ●
                ●
                    ●
                     ●
                     ●
                     ●
                      ●
                      ●
                      ●
                      ●
                      ●
                       ●
                           ●
                           ●
                           ●
                           ●
                           ●
                           ●
                            ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                       ●
                                       ●
                                       ●
                                       ●
                                       ●
                                        ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Class 1 scores (sorted): [0.0, 0.10360140577131186, 0.11432931004379969, 0.11456271493981383, 0.11578640893929008, 0.21299446581638362, 0.21810772009707016, 0.22271856500957357, 0.2240092426021197, 0.22421643087667023, 0.22801492741797702, 0.23482666935375884, 0.3249119947528853, 0.32559942912736906, 0.33024347629285106, 0.3373496624497327, 0.3385687935026954, 0.41374531907866247, 0.4329190257970693, 0.4349168610369322, 0.43995093046489886, 0.4415718642360163, 0.44421336869533334, 0.4505832372483988, 0.4527721099130885, 0.45806120739797557, 0.46786669643494827, 0.547951937701255, 0.5493271385003113, 0.5506063044642326, 0.5531777708484364, 0.5562627913154764, 0.5580160232420784, 0.5608260051537706, 0.6622019018262424, 0.6641612745920475, 0.6642428734899561, 0.6648940803767306, 0.6665585063417838, 0.6735198589056716, 0.6737334640987785, 0.7715284097827202, 0.7737061051480313, 0.7762576551330229, 0.7773788411884629, 0.7776390337834486, 0.7793316139055324, 0.7800391072548793, 0.7814970997925095, 0.7823378004368201, 0.7877309983896374, 0.7898341472363178, 0.8000470026898868, 0.8851035444528033, 0.8853433150581735, 0.8855568499922921, 0.8857098885025821, 0.8897847536826424, 0.8909021318716329, 0.8915242376856428, 0.8920495162909213, 0.8932763385607295, 0.8952659040853191, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10415526042245317, 0.10521673943396975, 0.10585772496970136, 0.11186704531995191, 0.11319600657184342, 0.11405332806148025, 0.11424377513536506, 0.11425571372814128, 0.11582408701770039, 0.11621474284447299, 0.1162911596457003, 0.12366553157549146, 0.20831189644933087, 0.21108458770122188, 0.2135476596646141, 0.21366833479446565, 0.21650613649457395, 0.21715731298194973, 0.21814373291633446, 0.21926667270957045, 0.21987720710257058, 0.22106718650789928, 0.22145839368137038, 0.22286041534165513, 0.22323309629570218, 0.22788188704563214, 0.22921466948507066, 0.3188139542662476, 0.3188932676271068, 0.3255042604204113, 0.3266532854156651, 0.3305513023515188, 0.33124871453608906, 0.3321784018043347, 0.3340216585877999, 0.3353075519117091, 0.33582142058549086, 0.33692591038798747, 0.34089114719636887, 0.3467098547283572, 0.3553924796339989, 0.43486371911257804, 0.4351014863949231, 0.43638103246508514, 0.4413427117825313, 0.44234869327073223, 0.44456984206416816, 0.45827524848506257, 0.46240674595671866, 0.46518206762557124, 0.4682882889602982, 0.47989372391078927, 0.5494388335864528, 0.5521283209107652, 0.5572551542874694, 0.5637905130609223, 0.5641796431559642, 0.6641523764002556, 0.6658790835910697, 0.6685843127465066, 0.7689339650569887]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.23106603494301137, 1: 0.0}
  Class 1 threshold: 0.0
  α=0.01: threshold=0.0000
  α=0.01: storing threshold=0.0000
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.33584762359974446, 1: 0.11456271493981383}
  Class 1 threshold: 0.11456271493981383
  α=0.05: threshold=0.1146
  α=0.05: storing threshold=0.1146
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.4427448457125305, 1: 0.22271856500957357}
  Class 1 threshold: 0.22271856500957357
  α=0.10: threshold=0.2227
  α=0.10: storing threshold=0.2227
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.5417247515149375, 1: 0.33024347629285106}
  Class 1 threshold: 0.33024347629285106
  α=0.20: threshold=0.3302
  α=0.20: storing threshold=0.3302
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.5586572882174689, 1: 0.4329190257970693}
  Class 1 threshold: 0.4329190257970693
  α=0.25: threshold=0.4329
  α=0.25: storing threshold=0.4329
HCD table has 752 rows
Using spatial coordinates for all 752 HCD points
Computing probabilities for 752 HCD points...
Probability matrix shape: (752, 2)
Export: Calibration class distribution: {0: 67, 1: 75}

Export: Class 1 calibration scores (75 points):
  Min: 0.0000
  Max: 1.0000
  Mean: 0.6257
  Std: 0.2843
  Percentiles: 25%=0.4339, 50%=0.6649, 75%=0.8856, 95%=1.0000, 99%=1.0000

Export: Class 0 calibration scores (67 points):
  Min: 0.0000
  Max: 0.7689
  Mean: 0.2960
  Std: 0.1847

Export: Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
     ●
     ●
     ●
     ●
          ●
          ●
           ●
           ●
           ●
           ●
           ●
                ●
                ●
                ●
                ●
                ●
                    ●
                     ●
                     ●
                     ●
                      ●
                      ●
                      ●
                      ●
                      ●
                       ●
                           ●
                           ●
                           ●
                           ●
                           ●
                           ●
                            ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                 ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                      ●
                                       ●
                                       ●
                                       ●
                                       ●
                                       ●
                                        ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                            ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
                                                  ●
0.0==================================================1.0

Export: Class 1 scores (sorted): [0.0, 0.10360140577131186, 0.11432931004379969, 0.11456271493981383, 0.11578640893929008, 0.21299446581638362, 0.21810772009707016, 0.22271856500957357, 0.2240092426021197, 0.22421643087667023, 0.22801492741797702, 0.23482666935375884, 0.3249119947528853, 0.32559942912736906, 0.33024347629285106, 0.3373496624497327, 0.3385687935026954, 0.41374531907866247, 0.4329190257970693, 0.4349168610369322, 0.43995093046489886, 0.4415718642360163, 0.44421336869533334, 0.4505832372483988, 0.4527721099130885, 0.45806120739797557, 0.46786669643494827, 0.547951937701255, 0.5493271385003113, 0.5506063044642326, 0.5531777708484364, 0.5562627913154764, 0.5580160232420784, 0.5608260051537706, 0.6622019018262424, 0.6641612745920475, 0.6642428734899561, 0.6648940803767306, 0.6665585063417838, 0.6735198589056716, 0.6737334640987785, 0.7715284097827202, 0.7737061051480313, 0.7762576551330229, 0.7773788411884629, 0.7776390337834486, 0.7793316139055324, 0.7800391072548793, 0.7814970997925095, 0.7823378004368201, 0.7877309983896374, 0.7898341472363178, 0.8000470026898868, 0.8851035444528033, 0.8853433150581735, 0.8855568499922921, 0.8857098885025821, 0.8897847536826424, 0.8909021318716329, 0.8915242376856428, 0.8920495162909213, 0.8932763385607295, 0.8952659040853191, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

Export: Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10415526042245317, 0.10521673943396975, 0.10585772496970136, 0.11186704531995191, 0.11319600657184342, 0.11405332806148025, 0.11424377513536506, 0.11425571372814128, 0.11582408701770039, 0.11621474284447299, 0.1162911596457003, 0.12366553157549146, 0.20831189644933087, 0.21108458770122188, 0.2135476596646141, 0.21366833479446565, 0.21650613649457395, 0.21715731298194973, 0.21814373291633446, 0.21926667270957045, 0.21987720710257058, 0.22106718650789928, 0.22145839368137038, 0.22286041534165513, 0.22323309629570218, 0.22788188704563214, 0.22921466948507066, 0.3188139542662476, 0.3188932676271068, 0.3255042604204113, 0.3266532854156651, 0.3305513023515188, 0.33124871453608906, 0.3321784018043347, 0.3340216585877999, 0.3353075519117091, 0.33582142058549086, 0.33692591038798747, 0.34089114719636887, 0.3467098547283572, 0.3553924796339989, 0.43486371911257804, 0.4351014863949231, 0.43638103246508514, 0.4413427117825313, 0.44234869327073223, 0.44456984206416816, 0.45827524848506257, 0.46240674595671866, 0.46518206762557124, 0.4682882889602982, 0.47989372391078927, 0.5494388335864528, 0.5521283209107652, 0.5572551542874694, 0.5637905130609223, 0.5641796431559642, 0.6641523764002556, 0.6658790835910697, 0.6685843127465066, 0.7689339650569887]
Export: Using confidence_level=0.990 for alpha=0.01
Export: Class 1 threshold: 0.0
Export: α=0.01: threshold=0.0000
Export: α=0.01: storing threshold=0.0000
Export: Using confidence_level=0.950 for alpha=0.05
Export: Class 1 threshold: 0.11456271493981383
Export: α=0.05: threshold=0.1146
Export: α=0.05: storing threshold=0.1146
Export: Using confidence_level=0.900 for alpha=0.10
Export: Class 1 threshold: 0.22271856500957357
Export: α=0.10: threshold=0.2227
Export: α=0.10: storing threshold=0.2227
Export: Using confidence_level=0.800 for alpha=0.20
Export: Class 1 threshold: 0.33024347629285106
Export: α=0.20: threshold=0.3302
Export: α=0.20: storing threshold=0.3302
Export: Using confidence_level=0.750 for alpha=0.25
Export: Class 1 threshold: 0.4329190257970693
Export: α=0.25: threshold=0.4329
Export: α=0.25: storing threshold=0.4329

Existing table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 180.0+ bytes
None

Existing table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m3             0.0  ...        0.328895        0.428518

[1 rows x 6 columns]

New data structure:
<class 'pandas.core.frame.DataFrame'>
Index: 1 entries, 0 to 0
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      1 non-null      object 
 1   threshold_0.01  1 non-null      float64
 2   threshold_0.05  1 non-null      float64
 3   threshold_0.10  1 non-null      float64
 4   threshold_0.20  1 non-null      float64
 5   threshold_0.25  1 non-null      float64
dtypes: float64(5), object(1)
memory usage: 56.0+ bytes
None

New data content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m4             0.0  ...        0.330243        0.432919

[1 rows x 6 columns]

Merged table structure:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 6 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   model_name      2 non-null      object 
 1   threshold_0.01  2 non-null      float64
 2   threshold_0.05  2 non-null      float64
 3   threshold_0.10  2 non-null      float64
 4   threshold_0.20  2 non-null      float64
 5   threshold_0.25  2 non-null      float64
dtypes: float64(5), object(1)
memory usage: 228.0+ bytes
None

Merged table content:
     model_name  threshold_0.01  ...  threshold_0.20  threshold_0.25
0  knn_model_m3             0.0  ...        0.328895        0.428518
1  knn_model_m4             0.0  ...        0.330243        0.432919

[2 rows x 6 columns]

Appended 1 rows to existing table 'conformal_thresholds'
measured_levels shape: (568,)
measured_class_labels shape: (568,)
measured_indices shape: (568,)
embedding_df columns: ['hcd_indx', 'umap_x', 'umap_y', 'selection', 'class']
embedding_df index range: 0 to 567
measured_indices range: 0 to 723
Level 0: level_mask shape: (568,), sum: 29
Level 1: level_mask shape: (568,), sum: 13
Level 2: level_mask shape: (568,), sum: 22
Level 3: level_mask shape: (568,), sum: 33
Level 4: level_mask shape: (568,), sum: 38
Level 5: level_mask shape: (568,), sum: 69
Level 6: level_mask shape: (568,), sum: 89
Level 7: level_mask shape: (568,), sum: 142
Level 8: level_mask shape: (568,), sum: 133

Starting Weighted KNN training process... using training dataset (435 points)

=== Starting Weighted KNN Training ===
Data source: training dataset
Total points: 435
Class distribution in training data: {0: 382, 1: 53}
K Min: 7
K Max: 11
K Step: 2
σ Min: 0.5
σ Max: 6
σ Step: 0.5
p: 2
Holdout size: 0.2
Feature space: UMAP Embedding
Distance metric: Euclidean
Normalize features: False (never normalized)
Using predefined calibration set: 133 points
Training set class distribution: {0: 382, 1: 53}
Calibration set class distribution: {0: 119, 1: 14}
Training Weighted KNN 1/36: k=7, σ=0.50, p=2
  K=7, σ=0.50, p=2: Calibration Acc=0.8947, Brier=0.0732
Training Weighted KNN 2/36: k=7, σ=1.00, p=2
  K=7, σ=1.00, p=2: Calibration Acc=0.8947, Brier=0.0718
Training Weighted KNN 3/36: k=7, σ=1.50, p=2
  K=7, σ=1.50, p=2: Calibration Acc=0.8947, Brier=0.0702
Training Weighted KNN 4/36: k=7, σ=2.00, p=2
  K=7, σ=2.00, p=2: Calibration Acc=0.8872, Brier=0.0702
Training Weighted KNN 5/36: k=7, σ=2.50, p=2
  K=7, σ=2.50, p=2: Calibration Acc=0.8872, Brier=0.0710
Training Weighted KNN 6/36: k=7, σ=3.00, p=2
  K=7, σ=3.00, p=2: Calibration Acc=0.8947, Brier=0.0717
Training Weighted KNN 7/36: k=7, σ=3.50, p=2
  K=7, σ=3.50, p=2: Calibration Acc=0.8947, Brier=0.0723
Training Weighted KNN 8/36: k=7, σ=4.00, p=2
  K=7, σ=4.00, p=2: Calibration Acc=0.8947, Brier=0.0726
Training Weighted KNN 9/36: k=7, σ=4.50, p=2
  K=7, σ=4.50, p=2: Calibration Acc=0.8947, Brier=0.0728
Training Weighted KNN 10/36: k=7, σ=5.00, p=2
  K=7, σ=5.00, p=2: Calibration Acc=0.8947, Brier=0.0729
Training Weighted KNN 11/36: k=7, σ=5.50, p=2
  K=7, σ=5.50, p=2: Calibration Acc=0.8947, Brier=0.0730
Training Weighted KNN 12/36: k=7, σ=6.00, p=2
  K=7, σ=6.00, p=2: Calibration Acc=0.8947, Brier=0.0731
Training Weighted KNN 13/36: k=9, σ=0.50, p=2
  K=9, σ=0.50, p=2: Calibration Acc=0.8947, Brier=0.0733
Training Weighted KNN 14/36: k=9, σ=1.00, p=2
  K=9, σ=1.00, p=2: Calibration Acc=0.8947, Brier=0.0719
Training Weighted KNN 15/36: k=9, σ=1.50, p=2
  K=9, σ=1.50, p=2: Calibration Acc=0.8947, Brier=0.0706
Training Weighted KNN 16/36: k=9, σ=2.00, p=2
  K=9, σ=2.00, p=2: Calibration Acc=0.8947, Brier=0.0713
Training Weighted KNN 17/36: k=9, σ=2.50, p=2
  K=9, σ=2.50, p=2: Calibration Acc=0.8947, Brier=0.0730
Training Weighted KNN 18/36: k=9, σ=3.00, p=2
  K=9, σ=3.00, p=2: Calibration Acc=0.8872, Brier=0.0746
Training Weighted KNN 19/36: k=9, σ=3.50, p=2
  K=9, σ=3.50, p=2: Calibration Acc=0.8872, Brier=0.0757
Training Weighted KNN 20/36: k=9, σ=4.00, p=2
  K=9, σ=4.00, p=2: Calibration Acc=0.8872, Brier=0.0765
Training Weighted KNN 21/36: k=9, σ=4.50, p=2
  K=9, σ=4.50, p=2: Calibration Acc=0.8872, Brier=0.0769
Training Weighted KNN 22/36: k=9, σ=5.00, p=2
  K=9, σ=5.00, p=2: Calibration Acc=0.8872, Brier=0.0772
Training Weighted KNN 23/36: k=9, σ=5.50, p=2
  K=9, σ=5.50, p=2: Calibration Acc=0.8872, Brier=0.0774
Training Weighted KNN 24/36: k=9, σ=6.00, p=2
  K=9, σ=6.00, p=2: Calibration Acc=0.8872, Brier=0.0775
Training Weighted KNN 25/36: k=11, σ=0.50, p=2
  K=11, σ=0.50, p=2: Calibration Acc=0.8947, Brier=0.0735
Training Weighted KNN 26/36: k=11, σ=1.00, p=2
  K=11, σ=1.00, p=2: Calibration Acc=0.8947, Brier=0.0722
Training Weighted KNN 27/36: k=11, σ=1.50, p=2
  K=11, σ=1.50, p=2: Calibration Acc=0.8947, Brier=0.0713
Training Weighted KNN 28/36: k=11, σ=2.00, p=2
  K=11, σ=2.00, p=2: Calibration Acc=0.8947, Brier=0.0727
Training Weighted KNN 29/36: k=11, σ=2.50, p=2
  K=11, σ=2.50, p=2: Calibration Acc=0.8947, Brier=0.0752
Training Weighted KNN 30/36: k=11, σ=3.00, p=2
  K=11, σ=3.00, p=2: Calibration Acc=0.8947, Brier=0.0775
Training Weighted KNN 31/36: k=11, σ=3.50, p=2
  K=11, σ=3.50, p=2: Calibration Acc=0.8947, Brier=0.0792
Training Weighted KNN 32/36: k=11, σ=4.00, p=2
  K=11, σ=4.00, p=2: Calibration Acc=0.8947, Brier=0.0803
Training Weighted KNN 33/36: k=11, σ=4.50, p=2
  K=11, σ=4.50, p=2: Calibration Acc=0.8947, Brier=0.0810
Training Weighted KNN 34/36: k=11, σ=5.00, p=2
  K=11, σ=5.00, p=2: Calibration Acc=0.8947, Brier=0.0815
Training Weighted KNN 35/36: k=11, σ=5.50, p=2
  K=11, σ=5.50, p=2: Calibration Acc=0.8947, Brier=0.0818
Training Weighted KNN 36/36: k=11, σ=6.00, p=2
  K=11, σ=6.00, p=2: Calibration Acc=0.8947, Brier=0.0820

=== Training Complete ===
Process started with PID: 53711
Processing training results
Computing conformal thresholds for selected model...
Calibration data shape: (133, 2)
Calibration labels shape: (133,)
Calibration class distribution: {0: 119, 1: 14}
Minimum samples per class: 14

Class 1 calibration scores (14 points):
  Min: 0.0927
  Max: 0.5559
  Mean: 0.2710
  Std: 0.1404
  Percentiles: 25%=0.1873, 50%=0.2076, 75%=0.3503, 95%=0.5521, 99%=0.5551

Class 0 calibration scores (119 points):
  Min: 0.0000
  Max: 0.6311
  Mean: 0.1011
  Std: 0.1212

Class 1 probability distribution (dot plot):
0.0==================================================1.0
    ●
     ●
         ●
         ●
         ●
          ●
          ●
          ●
              ●
              ●
                  ●
                  ●
                           ●
                           ●
0.0==================================================1.0

Class 1 scores (sorted): [0.09269228054642696, 0.10008818687428152, 0.18099770402592438, 0.18673377646278494, 0.18887730776664438, 0.20195765341923413, 0.20401679072772466, 0.21113983545943926, 0.2876101624807337, 0.29063030653911764, 0.3701460249587851, 0.37372839651100354, 0.5499905543028021, 0.5559180839435858]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0691715169482092, 0.07062514487455979, 0.07080306076392423, 0.07240632429776664, 0.07277049851078475, 0.07419960372977591, 0.07539884236459393, 0.07540210521959767, 0.07574018126290524, 0.07811127585616102, 0.07860246431717548, 0.07864271460195563, 0.07937592315267873, 0.07967513972871147, 0.08001238410158758, 0.08008232832024206, 0.08058098363989576, 0.08064719765708202, 0.08083180622006285, 0.08100484474932561, 0.08431548783221741, 0.08732775181012975, 0.08892825408597249, 0.09000178303220532, 0.09053745071362873, 0.09164680853748378, 0.09364881454097672, 0.0938680226772557, 0.0938901565381662, 0.09420486242929359, 0.09487795854780999, 0.09682479959378684, 0.09698497842468186, 0.10068045130031933, 0.10094255903242305, 0.1011309240773821, 0.10165765677421733, 0.10562959346877875, 0.10839989160206878, 0.10988095961943171, 0.16231786101013146, 0.16637617270264707, 0.16756627853259526, 0.16818796476079736, 0.170375521860898, 0.17075713502657253, 0.17116444689818355, 0.17789303703638465, 0.17933535123644587, 0.1801163466943988, 0.1805361819636036, 0.18199927882555492, 0.18338477911993128, 0.18375617020314955, 0.19429380464127355, 0.19827869736019144, 0.19875150359644136, 0.20995579755194796, 0.23916879352932985, 0.2571897425361694, 0.25912749326853934, 0.2600583088242489, 0.2744555306182245, 0.27667399624484923, 0.28696914151750597, 0.3365181309951424, 0.36300409913670706, 0.37315446592289325, 0.37431164658424954, 0.3913499068342712, 0.47108808693474363, 0.5300774619511159, 0.6311395026675032]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.469922538048884, 1: 0.09269228054642696}
  Class 1 threshold: 0.09269228054642696
  α=0.01: threshold=0.0927
  α=0.01: storing threshold=0.0927
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.6268455340771066, 1: 0.09269228054642696}
  Class 1 threshold: 0.09269228054642696
  α=0.05: threshold=0.0927
  α=0.05: storing threshold=0.0927
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.739941691175751, 1: 0.10008818687428152}
  Class 1 threshold: 0.10008818687428152
  α=0.10: threshold=0.1001
  α=0.10: storing threshold=0.1001
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.8198836533056012, 1: 0.18099770402592438}
  Class 1 threshold: 0.18099770402592438
  α=0.20: threshold=0.1810
  α=0.20: storing threshold=0.1810
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.8318120352392027, 1: 0.18673377646278494}
  Class 1 threshold: 0.18673377646278494
  α=0.25: threshold=0.1867
  α=0.25: storing threshold=0.1867
Computing conformal thresholds for selected model...
Calibration data shape: (133, 2)
Calibration labels shape: (133,)
Calibration class distribution: {0: 119, 1: 14}
Minimum samples per class: 14

Class 1 calibration scores (14 points):
  Min: 0.0170
  Max: 0.6524
  Mean: 0.3345
  Std: 0.1789
  Percentiles: 25%=0.2411, 50%=0.3093, 75%=0.4616, 95%=0.6104, 99%=0.6440

Class 0 calibration scores (119 points):
  Min: 0.0000
  Max: 0.6055
  Mean: 0.0772
  Std: 0.1426

Class 1 probability distribution (dot plot):
0.0==================================================1.0
●
   ●
       ●
           ●
             ●
              ●
              ●
               ●
                  ●
                     ●
                       ●
                         ●
                             ●
                                ●
0.0==================================================1.0

Class 1 scores (sorted): [0.01703376401120981, 0.07145832064247586, 0.15156134451995623, 0.23130535062751642, 0.27056881289010076, 0.2909984221710743, 0.2987058417916128, 0.31990496465333246, 0.37356214237904795, 0.4398629981219492, 0.46882865588963973, 0.5097144694582386, 0.5878027873924281, 0.652370446384317]
Class 0 scores (sorted): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0043749511891445336, 0.004386800968003652, 0.004930258669289856, 0.005587434274577172, 0.005774948333608115, 0.006219915187223017, 0.006299801611697264, 0.006688776101113538, 0.00680901675331301, 0.007144450958443387, 0.007296017826685067, 0.007433637733527845, 0.007461688552620078, 0.00763155328985494, 0.007768028772536205, 0.008255803918995266, 0.00844440283760858, 0.008868997725004971, 0.010671300061451516, 0.011230102517136664, 0.011407628719586695, 0.013128014906173036, 0.013830902224483984, 0.014966725540516526, 0.016279413604815635, 0.016565325713757715, 0.016819668540239082, 0.017168983066629152, 0.018959729140440477, 0.02072715126848126, 0.021471548466229028, 0.022427134547406694, 0.0226271640833599, 0.023273770821623397, 0.02335212315521261, 0.02354644023377609, 0.02498988841678372, 0.025154513650356203, 0.02601700335518871, 0.037156446689184455, 0.041677574680334464, 0.04362778972258919, 0.047792973892156955, 0.047860782122527167, 0.05437755124195004, 0.05693602787591636, 0.0827675130339196, 0.10362211305656407, 0.1159506612399865, 0.13359155600209197, 0.15210267815191253, 0.18251334862019894, 0.19338094214837054, 0.1974049155546472, 0.22835668491767364, 0.25243959770659874, 0.266689983904201, 0.27138296762367725, 0.2778368068658007, 0.3111902635616789, 0.32290632509416234, 0.32513871997339044, 0.34120936711155575, 0.37831750608971637, 0.3869565178472044, 0.40825140335201077, 0.41489704308152064, 0.4255520393097594, 0.43057898111372234, 0.43437124799571847, 0.5144511119766864, 0.5857948538691388, 0.6055432326640902]
  Using confidence_level=0.990 for alpha=0.01
  All thresholds: {0: 0.4142051461308612, 1: 0.01703376401120981}
  Class 1 threshold: 0.01703376401120981
  α=0.01: threshold=0.0170
  α=0.01: storing threshold=0.0170
  Using confidence_level=0.950 for alpha=0.05
  All thresholds: {0: 0.5744479606902407, 1: 0.01703376401120981}
  Class 1 threshold: 0.01703376401120981
  α=0.05: threshold=0.0170
  α=0.05: storing threshold=0.0170
  Using confidence_level=0.900 for alpha=0.10
  All thresholds: {0: 0.6748612800266096, 1: 0.07145832064247586}
  Class 1 threshold: 0.07145832064247586
  α=0.10: threshold=0.0715
  α=0.10: storing threshold=0.0715
  Using confidence_level=0.800 for alpha=0.20
  All thresholds: {0: 0.8664084439979081, 1: 0.15156134451995623}
  Class 1 threshold: 0.15156134451995623
  α=0.20: threshold=0.1516
  α=0.20: storing threshold=0.1516
  Using confidence_level=0.750 for alpha=0.25
  All thresholds: {0: 0.9521392178774729, 1: 0.23130535062751642}
  Class 1 threshold: 0.23130535062751642
  α=0.25: threshold=0.2313
  α=0.25: storing threshold=0.2313
