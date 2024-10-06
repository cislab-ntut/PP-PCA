import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from PPPCA import (
    share_generation,
    reconstruct_secret,
    share_centered_data,
    share_covariance_matrix,
    secret_sharing_power_method,
    secret_sharing_eigenshift,
    process_eigenvalue,
    process_eigenvector,
)

# 文件路徑與對應的因子
file_paths = ["iris.csv", "wine.csv", "diabetes.csv"]  # 替換為您的文件路徑列表
file_factors = {"iris.csv": 6000.0, "wine.csv": 1000.0, "diabetes.csv": 1000.0}
file_iteration = {"iris.csv": 18, "wine.csv": 60, "diabetes.csv": 60}
all_normalized_mae_values = []
all_normalized_rmse_values = []
all_normalized_mae_vectors = []
all_normalized_rmse_vectors = []

for file_path in file_paths:
    # 讀取CSV檔案
    data = pd.read_csv(file_path).values.astype(np.float64)
    N, n_dim = data.shape

    print(f"檔案: {file_path}")

    # 獲取對應的因子
    FACTOR = file_factors[file_path]

    # 進行數據處理：乘以因子和四捨五入
    processed_data = np.around(data * FACTOR, decimals=3)
    processed_data_int = processed_data.astype(np.int64)

    # 產生shares
    shares1, shares2 = [], []
    for row in processed_data_int:
        share1_row, share2_row = [], []
        for value in row:
            share1, share2 = share_generation(int(value))
            share1_row.append(share1)
            share2_row.append(share2)
        shares1.append(share1_row)
        shares2.append(share2_row)

    # 計算shares的平均值和中心化數據
    mean_shares1, mean_shares2, centered_shares1, centered_shares2 = (
        share_centered_data(shares1, shares2, n_dim, N)
    )

    # 計算shares的共變數矩陣
    cov_shares1, cov_shares2 = share_covariance_matrix(
        centered_shares1, centered_shares2, n_dim, N
    )

    # 初始化特徵向量的shares
    vec_shares1, vec_shares2 = [share_generation(FACTOR)[0] for _ in range(n_dim)], [
        share_generation(FACTOR)[1] for _ in range(n_dim)
    ]

    B = 1  # 泰勒展開基點
    o = 10  # 泰勒展開的階數
    tau = 10  # 迭代次數

    max_iteration = file_iteration[file_path]
    tolerable_error = 6

    eigenvalues = []
    eigenvectors = []

    for _ in range(3):  # 假設要找出三個主成分
        lambda_new_1, lambda_new_2, new_vector1, new_vector2 = (
            secret_sharing_power_method(
                cov_shares1,
                cov_shares2,
                vec_shares1,
                vec_shares2,
                tolerable_error,
                max_iteration,
                B,
                o,
                tau,
            )
        )

        eigenvalues.append(reconstruct_secret(lambda_new_1, lambda_new_2))
        eigenvectors.append(
            [
                reconstruct_secret(new_vector1[i], new_vector2[i])
                for i in range(len(new_vector1))
            ]
        )

        cov_shares1, cov_shares2 = secret_sharing_eigenshift(
            cov_shares1,
            cov_shares2,
            [lambda_new_1],
            [lambda_new_2],
            new_vector1,
            new_vector2,
        )

    # 將特徵值和特徵向量進行排序
    eigenvalues, eigenvectors = zip(
        *sorted(zip(eigenvalues, eigenvectors), reverse=True)
    )

    # 處理特徵值和特徵向量
    processed_eigenvalues = np.array([process_eigenvalue(ev) for ev in eigenvalues])
    processed_eigenvectors = np.array([process_eigenvector(ev) for ev in eigenvectors])

    # 計算資料的平均值
    mean_values = np.mean(processed_data, axis=0)

    # 中心化資料
    centered_data = processed_data - mean_values

    # 計算共變數矩陣
    cov_matrix = np.cov(centered_data, rowvar=False)

    # 使用PCA計算特徵值和特徵向量
    pca = PCA(n_components=min(n_dim, 3))  # 取最小值，確保不會超過數據維度
    pca.fit(processed_data)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    # 計算特徵值的 MAE 和 RMSE
    mae_values = np.mean(np.abs(processed_eigenvalues - eigenvalues[:3]))
    rmse_values = np.sqrt(np.mean((processed_eigenvalues - eigenvalues[:3]) ** 2))

    # print(f"特徵值的 MAE: {mae_values}")
    # print(f"特徵值的 RMSE: {rmse_values}")

    # 計算特徵值的相對 MAE 和 RMSE
    value_data_range = np.max(eigenvalues[:3]) - np.min(eigenvalues[:3])
    relative_mae_values = mae_values / value_data_range
    relative_rmse_values = rmse_values / value_data_range

    # print(f"特徵值的相對 MAE: {relative_mae_values}")
    # print(f"特徵值的相對 RMSE: {relative_rmse_values}")

    # 計算特徵值的範圍，並進行標準化
    min_eigenvalue = np.min(eigenvalues[:3])
    max_eigenvalue = np.max(eigenvalues[:3])

    # 將明文的特徵值和隱私保護的特徵值標準化到相同的範圍 [0, 1]
    normalized_plaintext_values = (eigenvalues[:3] - min_eigenvalue) / (
        max_eigenvalue - min_eigenvalue
    )

    normalized_privacy_preserving_values = (processed_eigenvalues - min_eigenvalue) / (
        max_eigenvalue - min_eigenvalue
    )

    # 計算標準化後的特徵值 MAE 和 RMSE
    normalized_mae_values = np.mean(
        np.abs(normalized_plaintext_values - normalized_privacy_preserving_values)
    )
    normalized_rmse_values = np.sqrt(
        np.mean(
            (normalized_plaintext_values - normalized_privacy_preserving_values) ** 2
        )
    )

    # print(f"標準化後的特徵值 MAE: {normalized_mae_values}")
    # print(f"標準化後的特徵值 RMSE: {normalized_rmse_values}")

    # 計算特徵向量的 MAE 和 RMSE
    vector_mae = np.mean(np.abs(processed_eigenvectors - eigenvectors))
    vector_rmse = np.sqrt(np.mean((processed_eigenvectors - eigenvectors) ** 2))

    # print(f"特徵向量的 MAE: {vector_mae}")
    # print(f"特徵向量的 RMSE: {vector_rmse}")

    # 計算特徵向量的相對 MAE 和 RMSE
    vector_data_range = np.max(eigenvectors) - np.min(eigenvectors)
    relative_vector_mae = vector_mae / vector_data_range
    relative_vector_rmse = vector_rmse / vector_data_range

    # print(f"特徵向量的相對 MAE: {relative_vector_mae}")
    # print(f"特徵向量的相對 RMSE: {relative_vector_rmse}")

    # 確保每個特徵向量單獨標準化
    normalized_plaintext_vector = np.zeros_like(eigenvectors)
    normalized_privacy_preserving_vector = np.zeros_like(processed_eigenvectors)

    # 對每一列進行標準化處理
    for i in range(eigenvectors.shape[1]):
        plaintext_min = np.min(eigenvectors[:, i])
        plaintext_max = np.max(eigenvectors[:, i])
        privacy_min = np.min(processed_eigenvectors[:, i])
        privacy_max = np.max(processed_eigenvectors[:, i])

        normalized_plaintext_vector[:, i] = (eigenvectors[:, i] - plaintext_min) / (
            plaintext_max - plaintext_min
        )
        normalized_privacy_preserving_vector[:, i] = (
            processed_eigenvectors[:, i] - privacy_min
        ) / (privacy_max - privacy_min)

    # 計算標準化後的 MAE 和 RMSE
    normalized_vector_mae = np.mean(
        np.abs(normalized_plaintext_vector - normalized_privacy_preserving_vector)
    )
    normalized_vector_rmse = np.sqrt(
        np.mean(
            (normalized_plaintext_vector - normalized_privacy_preserving_vector) ** 2
        )
    )

    # print(f"標準化後的向量 MAE: {normalized_vector_mae}")
    # print(f"標準化後的向量 RMSE: {normalized_vector_rmse}")

    # 添加到列表中
    all_normalized_mae_values.append(normalized_mae_values)
    all_normalized_rmse_values.append(normalized_rmse_values)
    all_normalized_mae_vectors.append(normalized_vector_mae)
    all_normalized_rmse_vectors.append(normalized_vector_rmse)

# 在最下面進行匯總輸出
print("三個資料集標準化後的特徵值 MAE:", all_normalized_mae_values)
print("三個資料集標準化後的特徵值 RMSE:", all_normalized_rmse_values)
print("三個資料集標準化後的特徵向量 MAE:", all_normalized_mae_vectors)
print("三個資料集標準化後的特徵向量 RMSE:", all_normalized_rmse_vectors)
