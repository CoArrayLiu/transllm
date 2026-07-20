import torch
try:
    from .metrics import All_Metrics
except ImportError:
    from metrics import All_Metrics
import json
import numpy as np
import os
import argparse
from collections import OrderedDict
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

def test(folder_path, mode="regression", mae_thresh=None, mape_thresh=0.0):
    grouped = OrderedDict()

    # Retrieve all JSON files from a folder and sort them by filename
    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")])

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            sample_index = int(i_data["id"].rsplit('_', 1)[-1])
            group = grouped.setdefault(sample_index, {"true": [], "pred": []})
            group["true"].append(np.array(i_data["y_in"]))
            group["pred"].append(np.array(i_data["st_pre_infolow"]))

    if not grouped:
        raise ValueError(f"No result samples found in {folder_path}")
    y_true_in_regionlist = [np.stack(group["true"], axis=-1) for group in grouped.values()]
    y_pred_in_regionlist = [np.stack(group["pred"], axis=-1) for group in grouped.values()]
    print('sample_windows', len(grouped))
    y_true_in = np.expand_dims(np.concatenate(y_true_in_regionlist, axis=-1),axis=0)
    y_pred_in = np.expand_dims(np.concatenate(y_pred_in_regionlist, axis=-1),axis=0)
    # y_true_in = np.stack(y_true_in_regionlist, axis=0)
    # y_pred_in = np.stack(y_pred_in_regionlist, axis=0)

    y_pred_in = np.abs(y_pred_in)
    print(y_true_in.shape, y_pred_in.shape)

    if mode == 'classification':
        test_classfication(y_true_in, y_pred_in)
    else:
        for t in range(y_true_in.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_in[:, t, ...], y_true_in[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_in, y_true_in, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))



def test_classfication(y_true_in, y_pred_in):

    for i in range(1):
        y_true = y_true_in
        y_pred = y_pred_in
        y_true[y_true > 1] = 1
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"MicroF1: {micro_f1:.2f}")
        print(f"MacroF1: {macro_f1:.2f}")
        print(f"f1 Score: {f1:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--dataset", choices=("SD", "SZ", "pems08", "pems03", "pems04", "urbanev"), required=True)
    parser.add_argument("--mode", choices=("regression", "classification"), default="regression")
    parser.add_argument("--mae_thresh", type=float, default=None)
    parser.add_argument("--mape_thresh", type=float, default=0.0)
    args = parser.parse_args()
    print(f"Evaluating {args.dataset} from {args.folder_path}")
    test(args.folder_path, args.mode, args.mae_thresh, args.mape_thresh)
