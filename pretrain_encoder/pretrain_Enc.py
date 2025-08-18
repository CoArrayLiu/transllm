import os
# 设置CUDA设备为单卡
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from joblib import Parallel, delayed
import torch
torch.set_num_threads(3)

from src.models.localgat2dataset import LocalGAT1
from src.engines.localgat_alldataset_engine import LocalGAT_Engine
from src.utils.dataloader import load_dataset,load_dataset_test, load_adj_from_numpy, get_dataset_info,load_dataset_sh,load_dataset_sh_baseline,load_dataset_sh_baseline_test
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from fastdtw import fastdtw
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset1', type=str, default='SD')
    parser.add_argument('--dataset2', type=str, default='shenzhen')
    parser.add_argument('--dataset3', type=str, default='shanghai')
    parser.add_argument('--dataset4', type=str, default='urbanev')
    parser.add_argument('--dataset5', type=str, default='pems08')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2021')
    parser.add_argument('--model_name', type=str, default='localgat_alldataset')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--bs', type=int, default=12)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=60)
    return parser  
def get_config():
    parser = get_public_config()
    parser.add_argument('--tpd', type=int, default=288, help='time per day')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--thres', type=float, default=0.6)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--time_stride', type=int, default=1)
    # parser.add_argument('--model_name', type=str, default='localgat_contxt_nofanguiyi')
    parser.add_argument('--node_information_path1', type=str, default='data/st_data/sd/sd_meta.csv')
    parser.add_argument('--node_information_path2', type=str, default='data/st_data/shenzhen/sz_meta.csv')
    parser.add_argument('--node_information_path4', type=str, default='data/st_data/urbanev/meta.csv')
    parser.add_argument('--lrate', type=float, default=5e-4)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './pretrain_encoder/experiments/{}/{}/'.format(args.model_name, 'alldataset')
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def compute_pair(i, j, data_mean):
    print(f"Computing ({i}, {j})")
    dist, _ = fastdtw(data_mean[i], data_mean[j], radius=6)
    return (i, j, dist)

def compute_distance_matrix(data_mean, node_num, n_jobs=8):
    dist_matrix = np.zeros((node_num, node_num))

    # 并行计算上三角
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pair)(i, j, data_mean)
        for i in range(node_num) for j in range(i, node_num)
    )

    # 填充对称矩阵
    for i, j, dist in results:
        dist_matrix[i][j] = dist
        dist_matrix[j][i] = dist

    return dist_matrix

def construct_se_matrix(data_path, args):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    if args.years=='2015':
        # data = ptr['empty_count']
        # # demand_count = ptr['demand_count']
        # sample_num, node_num = data.shape

        # data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
        # data_mean = data_mean.T
        
        # dist_matrix = compute_distance_matrix(data_mean, node_num, n_jobs=8)
        # # for i in range(node_num):
        # #     for j in range(i, node_num):
        # #         dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        # # for i in range(node_num):
        # #     for j in range(i):
        # #         dist_matrix[i][j] = dist_matrix[j][i]

        # mean = np.mean(dist_matrix)
        # std = np.std(dist_matrix)
        # dist_matrix = (dist_matrix - mean) / std
        # dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
        # dtw_matrix = np.zeros_like(dist_matrix)
        # dtw_matrix[dist_matrix > args.thres] = 1
        # save_path = os.path.join(data_path, "cached_empty_matrix.npy")
        # # save_path = os.path.join(data_path, "cached_demand_matrix.npy")
        # np.save(save_path, dtw_matrix)

        data = ptr['waiting_count']
        # demand_count = ptr['demand_count']
        sample_num, node_num = data.shape

        data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
        data_mean = data_mean.T
        
        dist_matrix = compute_distance_matrix(data_mean, node_num, n_jobs=8)

        mean = np.mean(dist_matrix)
        std = np.std(dist_matrix)
        dist_matrix = (dist_matrix - mean) / std
        dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
        dtw_matrix = np.zeros_like(dist_matrix)
        dtw_matrix[dist_matrix > args.thres] = 1
        # save_path = os.path.join(data_path, "cached_empty_matrix.npy")
        save_path = os.path.join(data_path, "cached_waiting_matrix.npy")
        np.save(save_path, dtw_matrix)

        return dtw_matrix
    else:
        data = ptr['data'][..., 0]
        sample_num, node_num = data.shape

        data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
        data_mean = data_mean.T
        
        dist_matrix = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in range(i, node_num):
                dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
                print(i,j)
        for i in range(node_num):
            for j in range(i):
                dist_matrix[i][j] = dist_matrix[j][i]

        mean = np.mean(dist_matrix)
        std = np.std(dist_matrix)
        dist_matrix = (dist_matrix - mean) / std
        dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
        dtw_matrix = np.zeros_like(dist_matrix)
        dtw_matrix[dist_matrix > args.thres] = 1
        save_path = os.path.join(data_path, "cached_dist_matrix.npy")
        np.save(save_path, dtw_matrix)
        return dtw_matrix
def read_node_information(args):
    file_path = args.node_information_path1
    df = pd.read_csv(file_path)
    selected_cols = ["Fwy", "Lanes", "Direction"]  # 筛选目标列
    df = df[selected_cols].copy()
    df["Fwy_main"] = df["Fwy"].str.split("-").str[0]
    # ---------------------- 2. 特征编码 ----------------------
    # 2.1 道路类型（Fwy）：独热编码
    encoder_fwy = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    fwy_encoded = encoder_fwy.fit_transform(df[["Fwy_main"]])  # 形状：(N, 类别数)

    # # 2.2 车道数（Lanes）：数值标准化
    # scaler_lanes = StandardScaler()
    # lanes_scaled = scaler_lanes.fit_transform(df[["Lanes"]])  # 形状：(N, 1)
    lanes_raw = df[["Lanes"]].values.astype(int)
    valid_directions = ["N", "S", "W", "E"]

    # 创建编码器（自动忽略未知方向）
    encoder_direction = OneHotEncoder(
        categories=[valid_directions],
        sparse_output=False,
        handle_unknown="ignore"
    )

    # 执行编码
    direction_encoded = encoder_direction.fit_transform(df[["Direction"]])

    # 将编码后的特征水平拼接
    final_features = np.hstack([fwy_encoded, lanes_raw, direction_encoded])
    return final_features

def read_node_information2(args):
    file_path = args.node_information_path2
    df = pd.read_csv(file_path)
    
    selected_cols = ["count", "fast_count", "slow_count"]  # 筛选目标列
    df = df[selected_cols].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)  # shape: [N, 3]

    # 转为 NumPy 数组返回
    final_features = np.hstack([scaled_features])
    return final_features
def read_node_information4(args):
    file_path = args.node_information_path4
    df = pd.read_csv(file_path)
    
    selected_cols = ["charge_count"]  # 筛选目标列
    df = df[selected_cols].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)  # shape: [N, 3]

    # 转为 NumPy 数组返回
    final_features = np.hstack([scaled_features])
    return final_features
def normalize_adj_mx(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), adj_mx),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))

def load_adj(args):
    nodes_feature1 = read_node_information(args)
    nodes_feature1 = torch.from_numpy(nodes_feature1).float().to(args.device)
    data_path, adj_path, node_num = get_dataset_info(args.dataset1)
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    sp_matrix = adj_mx + np.transpose(adj_mx)
    rows, cols = np.where(sp_matrix)  # 提取非零元素的行列索引
    sp_matrix = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num
    sp_matrix = sp_matrix.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix = sp_matrix.reshape(2,-1)


    nodes_feature2 = read_node_information2(args)
    nodes_feature2 = torch.from_numpy(nodes_feature2).float().to(args.device)
    # se_matrix = construct_se_matrix(data_path, args)
    se_matrix =np.load(os.path.join(data_path, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix)  # 提取非零元素的行列索引
    se_matrix = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix = se_matrix.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix = se_matrix.reshape(2,-1)

    data_path2, adj_path2, node_num2 = get_dataset_info(args.dataset2)
    sp_matrix2 = load_adj_from_numpy(adj_path2)
    rows, cols = np.where(sp_matrix2)  # 提取非零元素的行列索引
    sp_matrix2 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num2
    sp_matrix2 = sp_matrix2.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix2 = sp_matrix2.reshape(2,-1)

    # se_matrix2 = construct_se_matrix(data_path2, args)
    se_matrix2 =np.load(os.path.join(data_path2, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix2)  # 提取非零元素的行列索引
    se_matrix2 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix2 = se_matrix2.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix2 = se_matrix2.reshape(2,-1)

    data_path3, adj_path3, node_num3 = get_dataset_info(args.dataset3)
    sp_matrix3 = load_adj_from_numpy(adj_path3)
    rows, cols = np.where(sp_matrix3)  # 提取非零元素的行列索引
    sp_matrix3 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num3
    sp_matrix3 = sp_matrix3.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix3 = sp_matrix3.reshape(2,-1)

    # se_matrix3 = construct_se_matrix(data_path3, args)
    se_matrix3 =np.load(os.path.join(data_path3, "cached_demand_matrix.npy"))
    rows, cols = np.where(se_matrix3)  # 提取非零元素的行列索引
    se_matrix3 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix3 = se_matrix3.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix3 = se_matrix3.reshape(2,-1)

    se_matrix3_2 =np.load(os.path.join(data_path3, "cached_waiting_matrix.npy"))
    rows, cols = np.where(se_matrix3_2)  # 提取非零元素的行列索引
    se_matrix3_2 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix3_2 = se_matrix3_2.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix3_2 = se_matrix3_2.reshape(2,-1)
    
    nodes_feature4 = read_node_information4(args)
    nodes_feature4 = torch.from_numpy(nodes_feature4).float().to(args.device)
    data_path4, adj_path4, node_num4 = get_dataset_info(args.dataset4)
    sp_matrix4 = load_adj_from_numpy(adj_path4)
    rows, cols = np.where(sp_matrix4)  # 提取非零元素的行列索引
    sp_matrix4 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num4
    sp_matrix4 = sp_matrix4.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix4 = sp_matrix4.reshape(2,-1)

    # se_matrix4 = construct_se_matrix(data_path4, args)
    se_matrix4 =np.load(os.path.join(data_path4, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix4)  # 提取非零元素的行列索引
    se_matrix4 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix4 = se_matrix4.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix4 = se_matrix4.reshape(2,-1)

    data_path5, adj_path5, node_num5 = get_dataset_info(args.dataset5)
    sp_matrix5 = load_adj_from_numpy(adj_path5)
    rows, cols = np.where(sp_matrix5)  # 提取非零元素的行列索引
    sp_matrix5 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num5
    sp_matrix5 = sp_matrix5.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix5 = sp_matrix5.reshape(2,-1)

    # se_matrix5 = construct_se_matrix(data_path5, args)
    se_matrix5 =np.load(os.path.join(data_path5, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix5)  # 提取非零元素的行列索引
    se_matrix5 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix5 = se_matrix5.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix5 = se_matrix5.reshape(2,-1)

    return nodes_feature1,sp_matrix,se_matrix,nodes_feature2,sp_matrix2,se_matrix2,sp_matrix3,se_matrix3,se_matrix3_2,nodes_feature4,sp_matrix4,se_matrix4,sp_matrix5,se_matrix5
def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    data_path1, adj_path1, node_num1 = get_dataset_info(args.dataset1)
    data_path2, adj_path2, node_num2 = get_dataset_info(args.dataset2)
    data_path3, adj_path3, node_num3 = get_dataset_info(args.dataset3)
    data_path5, adj_path5, node_num5 = get_dataset_info(args.dataset4)
    data_path6, adj_path6, node_num6 = get_dataset_info(args.dataset5)
    nodes_feature1,sp_matrix1,se_matrix1,nodes_feature2,sp_matrix2,se_matrix2,sp_matrix3,se_matrix3,se_matrix4,nodes_feature5,sp_matrix5,se_matrix5,sp_matrix6,se_matrix6=load_adj(args)

    dataloader1, scaler1 = load_dataset(data_path1, args, logger)
    dataloader2, scaler2 = load_dataset(data_path2, args, logger)
    dataloader3, scaler3,dataloader4, scaler4 = load_dataset_sh(data_path3, args, logger)
    dataloader5, scaler5 = load_dataset(data_path5, args, logger)
    dataloader6, scaler6 = load_dataset(data_path6, args, logger)
    model = LocalGAT1(input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   device=args.device,
                   hidden_dim=args.hidden_dim,
                   time_stride=args.time_stride
                   )
    # model_path = os.path.join('./pretrain_encoder/experiments/localgat_3dataset_new/3dataset/final_model_s2023.pt')
    # if os.path.exists(model_path):
    #     print(f"找到预训练模型: {model_path}")
    #     # 加载模型参数
    #     model.load_state_dict(torch.load(model_path, map_location=args.device))
    #     print("模型加载成功!")

    model.to(args.device)
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    engine = LocalGAT_Engine(device=device,
                           model=model,
                           dataloader1=dataloader1,
                           scaler1=scaler1,
                           dataloader2=dataloader2,
                           scaler2=scaler2,
                           dataloader3=dataloader3,
                           scaler3=scaler3,
                           dataloader4=dataloader4,
                           scaler4=scaler4,
                           dataloader5=dataloader5,
                           scaler5=scaler5,
                           dataloader6=dataloader6,
                           scaler6=scaler6,
                           sampler=None,
                           loss_fn=loss_fn,
                           lrate=args.lrate,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           clip_grad_value=args.clip_grad_value,
                           start_epoch = args.start_epochs,
                           max_epochs=args.max_epochs,
                           patience=args.patience,
                           log_dir=log_dir,
                           logger=logger,
                           seed=args.seed
                           )

    if args.mode == 'train':
        engine.train(sp_matrix1,se_matrix1,nodes_feature1,sp_matrix2,se_matrix2,nodes_feature2,sp_matrix3,se_matrix3,se_matrix4,sp_matrix5,se_matrix5,nodes_feature5,sp_matrix6,se_matrix6)
    else:
        engine.evaluate(args.mode,sp_matrix1,se_matrix1,nodes_feature1,sp_matrix2,se_matrix2,nodes_feature2,sp_matrix3,se_matrix3,se_matrix4,sp_matrix5,se_matrix5,nodes_feature5,sp_matrix6,se_matrix6)


if __name__ == "__main__":
    main()