import numpy as np
import sympy as sp
import time
import pandas as pd

PRIME = 524287


def share_generation(value):
    share1 = int(np.random.randint(0, PRIME))  # 轉換為 Python int
    share2 = (value - share1) % PRIME  # Python int 直接計算
    return share1, share2


def reconstruct_secret(share1, share2):
    return (share1 + share2) % PRIME  # 使用 Python int


def _extended_gcd(a, b):
    x, last_x = 0, 1
    y, last_y = 1, 0

    while b != 0:
        quot = a // b
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y

    return last_x, last_y


def _divmod(num, den, p):
    invert, _ = _extended_gcd(den, p)
    return num * invert % p


def multiply_shares(share1_a, share2_a, share1_b, share2_b, a, b):
    c = (a * b) % PRIME

    share_a1, share_a2 = share_generation(a)
    share_b1, share_b2 = share_generation(b)
    share_c1, share_c2 = share_generation(c)

    alpha_share1 = (share1_a - share_a1) % PRIME
    alpha_share2 = (share2_a - share_a2) % PRIME
    beta_share1 = (share1_b - share_b1) % PRIME
    beta_share2 = (share2_b - share_b2) % PRIME

    alpha = reconstruct_secret(alpha_share1, alpha_share2)
    beta = reconstruct_secret(beta_share1, beta_share2)

    term1_share1 = (alpha * share_b1) % PRIME
    term1_share2 = (alpha * share_b2) % PRIME
    term2_share1 = (beta * share_a1) % PRIME
    term2_share2 = (beta * share_a2) % PRIME
    alpha_beta_share1, alpha_beta_share2 = share_generation(alpha * beta)

    new_share1 = (share_c1 + term1_share1 + term2_share1 + alpha_beta_share1) % PRIME
    new_share2 = (share_c2 + term1_share2 + term2_share2 + alpha_beta_share2) % PRIME

    return new_share1, new_share2


def share_centered_data(shares1, shares2, n_dim, N):

    mean_shares1 = [
        ((sum(share) % PRIME) * _divmod(1, N, PRIME)) % PRIME for share in zip(*shares1)
    ]
    mean_shares2 = [
        ((sum(share) % PRIME) * _divmod(1, N, PRIME)) % PRIME for share in zip(*shares2)
    ]

    centered_shares1 = []
    centered_shares2 = []

    for i in range(N):
        centered_row1 = []
        centered_row2 = []
        for j in range(n_dim):
            centered_value1 = (shares1[i][j] - mean_shares1[j]) % PRIME
            centered_value2 = (shares2[i][j] - mean_shares2[j]) % PRIME

            centered_row1.append(centered_value1)
            centered_row2.append(centered_value2)

        centered_shares1.append(centered_row1)
        centered_shares2.append(centered_row2)

    return mean_shares1, mean_shares2, centered_shares1, centered_shares2


def share_covariance_matrix(centered_shares1, centered_shares2, n_dim, N):
    transposed_centered_shares1 = list(zip(*centered_shares1))
    transposed_centered_shares2 = list(zip(*centered_shares2))

    cov_shares1 = [[0] * n_dim for _ in range(n_dim)]
    cov_shares2 = [[0] * n_dim for _ in range(n_dim)]

    for i in range(n_dim):
        for j in range(n_dim):
            for k in range(N):
                product_share1, product_share2 = multiply_shares(
                    transposed_centered_shares1[i][k],
                    transposed_centered_shares2[i][k],
                    centered_shares1[k][j],
                    centered_shares2[k][j],
                    np.random.randint(0, PRIME),  # Python int
                    np.random.randint(0, PRIME),
                )
                cov_shares1[i][j] = (cov_shares1[i][j] + product_share1) % PRIME
                cov_shares2[i][j] = (cov_shares2[i][j] + product_share2) % PRIME

            inv_n_minus_1 = _divmod(1, N - 1, PRIME)
            cov_shares1[i][j] = (cov_shares1[i][j] * inv_n_minus_1) % PRIME
            cov_shares2[i][j] = (cov_shares2[i][j] * inv_n_minus_1) % PRIME

    return cov_shares1, cov_shares2


def add_shares(share1_1, share1_2, share2_1, share2_2):
    """假設這是一個秘密分享的加法函數"""
    sum_share1 = (share1_1 + share2_1) % PRIME
    sum_share2 = (share1_2 + share2_2) % PRIME
    return sum_share1, sum_share2


def taylor_expansion_sqrt_inv(norm_squared, B, o):
    # print("TAY")
    x = sp.symbols("x")
    func = 1 / sp.sqrt(x)
    taylor_series = sp.series(func, x, B, o).removeO().subs(x, value)
    return taylor_series.evalf()
    # x = sp.symbols("x")
    # func = 1 / sp.sqrt(1 - B * x)

    # # 假設 norm_squared 是 NumPy 陣列，提取其中的單個值
    # if isinstance(norm_squared, np.ndarray):
    #     if norm_squared.size == 1:
    #         value = norm_squared.item()  # 提取單個標量值
    #     else:
    #         raise ValueError("Expected a single value in the array.")
    # else:
    #     value = norm_squared

    # # 進行泰勒展開並替換
    # taylor_series = sp.series(func, x, 0, o).removeO().subs(x, value)

    # return taylor_series


def secret_sharing_inv_sqrt(x0_1, x0_2, x_prime_1, x_prime_2, tau):
    # print("INV")
    S1, S2 = x_prime_1, x_prime_2
    for _ in range(tau):
        S1, S2 = multiply_shares(
            S1,
            S2,
            S1,
            S2,
            np.random.randint(0, PRIME),
            np.random.randint(0, PRIME),
        )
        S1, S2 = multiply_shares(
            x0_1,
            x0_2,
            S1,
            S2,
            np.random.randint(0, PRIME),
            np.random.randint(0, PRIME),
        )
        TS1 = (3 - S1) % PRIME
        TS2 = (3 - S2) % PRIME  # TS = 3-xy^2

        HS1, HS2 = multiply_shares(  # HS= 1/2y
            _divmod(1, 2, PRIME),
            _divmod(1, 2, PRIME),
            x_prime_1,
            x_prime_2,
            np.random.randint(0, PRIME),
            np.random.randint(0, PRIME),
        )
        FS1, FS2 = multiply_shares(
            HS1,
            HS2,
            TS1,
            TS2,
            np.random.randint(0, PRIME),
            np.random.randint(0, PRIME),
        )
        x_prime_1, x_prime_2 = FS1, FS2
    return x_prime_1, x_prime_2


def secure_compare(
    new_lambda1, new_lambda2, prev_lambda1, prev_lambda2, tolerable_error
):
    # print("SC")
    # 產生tolerable_error的分享
    tolerable_error_share1, tolerable_error_share2 = share_generation(tolerable_error)

    while True:
        # 產生隨機值R並生成其分享
        R = np.random.randint(0, PRIME)
        R_share1, R_share2 = share_generation(R)

        # 計算s和h
        u1 = (new_lambda1 - prev_lambda1) % PRIME
        u2 = (new_lambda2 - prev_lambda2) % PRIME
        s1 = (u1 + R_share1) % PRIME
        s2 = (u2 + R_share2) % PRIME
        h1 = (tolerable_error_share1 + R_share1) % PRIME
        h2 = (tolerable_error_share2 + R_share2) % PRIME

        u = reconstruct_secret(u1, u2)

        # 檢查s1和h1是否滿足條件
        if (u + R < PRIME) and (tolerable_error_share1 + R < PRIME):
            # print(f"R:{R}")
            # print(
            #     f"recover u:{reconstruct_secret(u1, u2)}, tolerable_error:{reconstruct_secret(tolerable_error_share1, tolerable_error_share2)}"
            # )
            break

    # 還原s和h
    s = reconstruct_secret(s1, s2)
    h = reconstruct_secret(h1, h2)
    # print(f"s:{s}, h:{h}")

    # 比較s和h
    if s < h:
        return True
    else:
        return False


def secret_sharing_power_method(
    cov_shares1,
    cov_shares2,
    vec_shares1,
    vec_shares2,
    tolerable_error,
    max_iteration,
    B,
    o,
    tau,
):
    # print("SS PW Process")
    n_dim = len(cov_shares1)
    prev_lambda1, prev_lambda2 = share_generation(6000)
    # print(f"prev_lambda:{reconstruct_secret(prev_lambda1, prev_lambda2)}")

    for iter_count in range(max_iteration):
        # 計算向量與矩陣乘積
        new_vec_shares1 = [0] * n_dim
        new_vec_shares2 = [0] * n_dim
        for i in range(n_dim):
            for j in range(n_dim):
                product_share1, product_share2 = multiply_shares(
                    vec_shares1[j],
                    vec_shares2[j],
                    cov_shares1[j][i],
                    cov_shares2[j][i],
                    np.random.randint(0, PRIME),
                    np.random.randint(0, PRIME),  # 使用新的隨機數
                )
                new_vec_shares1[i] = (new_vec_shares1[i] + product_share1) % PRIME
                new_vec_shares2[i] = (new_vec_shares2[i] + product_share2) % PRIME

        # 計算特徵值
        new_lambda1, new_lambda2 = 0, 0
        for i in range(n_dim):
            product_share1, product_share2 = multiply_shares(
                new_vec_shares1[i],
                new_vec_shares2[i],
                vec_shares1[i],
                vec_shares2[i],
                *share_generation(0),  # 使用新的隨機數
            )
            new_lambda1 = (new_lambda1 + product_share1) % PRIME
            new_lambda2 = (new_lambda2 + product_share2) % PRIME

        # print(f"new_lambda:{reconstruct_secret(new_lambda1, new_lambda2)}")

        # 計算向量的平方和
        norm_squared1, norm_squared2 = 0, 0
        for i in range(n_dim):
            product_share1, product_share2 = multiply_shares(
                new_vec_shares1[i],
                new_vec_shares2[i],
                new_vec_shares1[i],
                new_vec_shares2[i],
                np.random.randint(0, PRIME),
                np.random.randint(0, PRIME),  # 使用新的隨機數
            )
            norm_squared1 = (norm_squared1 + product_share1) % PRIME
            norm_squared2 = (norm_squared2 + product_share2) % PRIME

        # 計算 1/sqrt(norm_squared) 的初始預估值
        taylor_inv_sqrt1 = taylor_expansion_sqrt_inv(norm_squared1, B, o)
        taylor_inv_sqrt2 = taylor_expansion_sqrt_inv(norm_squared2, B, o)

        # 使用泰勒展開式的初始預估值進行單位化
        inv_sqrt_norm_squared1, inv_sqrt_norm_squared2 = secret_sharing_inv_sqrt(
            norm_squared1, norm_squared2, taylor_inv_sqrt1, taylor_inv_sqrt2, tau
        )

        # 更新向量
        for i in range(n_dim):
            new_vec_shares1[i], new_vec_shares2[i] = multiply_shares(
                new_vec_shares1[i],
                new_vec_shares2[i],
                inv_sqrt_norm_squared1,
                inv_sqrt_norm_squared2,
                np.random.randint(0, PRIME),
                np.random.randint(0, PRIME),
            )

        # 檢查收斂
        if secure_compare(
            new_lambda1, new_lambda2, prev_lambda1, prev_lambda2, tolerable_error
        ):
            print(f"Get out iteration ! Number {iter_count} Stop!")
            break

        # 更新特徵向量和特徵值
        vec_shares1 = new_vec_shares1[:]
        vec_shares2 = new_vec_shares2[:]
        prev_lambda1, prev_lambda2 = new_lambda1, new_lambda2

    # print(f"Iteration Number = {iter_count+1} End the computation")
    return new_lambda1, new_lambda2, new_vec_shares1, new_vec_shares2


def secret_sharing_eigenshift(
    cov_shares1, cov_shares2, lambda_shares1, lambda_shares2, vec_shares1, vec_shares2
):
    # print("SSES Process")
    n_dim = len(cov_shares1)
    lambda_share1 = lambda_shares1[0]
    lambda_share2 = lambda_shares2[0]
    vec_shares1 = np.array(vec_shares1)
    vec_shares2 = np.array(vec_shares2)

    shift_cov1 = [[0] * n_dim for _ in range(n_dim)]
    shift_cov2 = [[0] * n_dim for _ in range(n_dim)]

    transposed_vector_shares1 = vec_shares1.reshape(-1, 1)
    transposed_vector_shares2 = vec_shares2.reshape(-1, 1)

    for i in range(n_dim):
        for j in range(n_dim):
            orig_share1, orig_share2 = cov_shares1[i][j], cov_shares2[i][j]
            term1_share1, term1_share2 = multiply_shares(
                lambda_share1,
                lambda_share2,
                transposed_vector_shares1[i],
                transposed_vector_shares2[i],
                np.random.randint(0, PRIME),  # 隨機數
                np.random.randint(0, PRIME),
            )
            term2_share1, term2_share2 = multiply_shares(
                term1_share1,
                term1_share2,
                vec_shares1[j],
                vec_shares2[j],
                np.random.randint(0, PRIME),  # 隨機數
                np.random.randint(0, PRIME),
            )
            shift_cov1[i][j] = (orig_share1 - term2_share1) % PRIME
            shift_cov2[i][j] = (orig_share2 - term2_share2) % PRIME
            # cov_shares1 = shift_cov1
            # cov_shares2 = shift_cov2

    return shift_cov1, shift_cov2


def process_eigenvalue(eigenvalue):
    if isinstance(eigenvalue, np.ndarray):
        return float(eigenvalue[0])
    return float(eigenvalue)


def process_eigenvector(eigenvector):
    return [float(x[0]) if isinstance(x, np.ndarray) else float(x) for x in eigenvector]


start_time = time.time()
np.random.seed(42)  # 設置隨機種子，確保每次運行結果一致


# 初始化數據
file_paths = ["iris.csv", "wine.csv", "diabetes.csv"]  # 替換為您的文件路徑列表
# 'wine.csv', 'diabetes.csv', 'iris.csv'

# 文件路徑與對應的因子
file_factors = {"iris.csv": 6000.0, "wine.csv": 1000.0, "diabetes.csv": 1000.0}
file_iteration = {"iris.csv": 20, "wine.csv": 50, "diabetes.csv": 45}

# 初始化一個空的列表來存儲輸出結果
output_lines = []

for file_path in file_paths:
    start_time = time.time()

    # 讀取CSV檔案
    data = pd.read_csv(file_path).values.astype(np.float64)
    N, n_dim = len(data), len(data[0])

    # 檢查讀入的資料
    print(f"檔案: {file_path}")
    # print(f"原始資料的前幾筆：") #（乘以 {FACTOR}）
    # print(data[:5])

    # 獲取對應的因子
    FACTOR = file_factors[file_path]

    # 進行數據處理：乘以因子和四捨五入
    processed_data = np.around(data * FACTOR, decimals=3)
    # 將資料轉換為整數格式
    processed_data_int = processed_data.astype(np.int64)

    # 顯示處理後的數據
    # print(f"處理後的資料的前幾筆（乘以 {FACTOR}）：")
    # print(processed_data[:5])

    # 產生shares
    shares1 = []
    shares2 = []

    for row in processed_data:
        share1_row = []
        share2_row = []
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

    # 還原平均值
    mean_values = [
        reconstruct_secret(mean_shares1[i], mean_shares2[i]) / FACTOR
        for i in range(n_dim)
    ]

    # 還原共變數矩陣
    cov_matrix = [
        [
            reconstruct_secret(cov_shares1[i][j], cov_shares2[i][j]) / (FACTOR * FACTOR)
            for j in range(n_dim)
        ]
        for i in range(n_dim)
    ]

    # 初始化特徵向量的 shares
    vec_shares1, vec_shares2 = [share_generation(FACTOR)[0] for _ in range(n_dim)], [
        share_generation(FACTOR)[1] for _ in range(n_dim)
    ]

    B = 1  # 泰勒展開基點
    o = 10  # 泰勒展開的階數
    tau = 10  # 迭代次數

    # 調整最大迭代次數
    max_iteration = file_iteration[file_path]

    # 更新tolerable_error
    tolerable_error = 6

    eigenvalues = []
    eigenvectors = []
    shift_covariances1 = []
    shift_covariances2 = []
    number_of_set = 3

    for _ in range(number_of_set):
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

        eigenvalues.append(reconstruct_secret(lambda_new_1, lambda_new_2) / FACTOR)
        eigenvectors.append(
            [
                reconstruct_secret(new_vector1[i], new_vector2[i]) / (FACTOR * FACTOR)
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

        shift_covariances1.append(cov_shares1)
        shift_covariances2.append(cov_shares2)

    # 將特徵值和特徵向量進行排序
    eigenvalues, eigenvectors = zip(
        *sorted(zip(eigenvalues, eigenvectors), reverse=True)
    )

    # 處理特徵值
    processed_eigenvalues = tuple(process_eigenvalue(ev) for ev in eigenvalues)

    # 處理特徵向量
    processed_eigenvectors = [process_eigenvector(ev) for ev in eigenvectors]

    end_time = time.time()

    # 顯示結果
    print("平均值:")
    print(mean_values)
    print("共變數矩陣:")
    for row in cov_matrix:
        print(row)
    print("特徵值:")
    print(processed_eigenvalues)
    print("特徵向量:")
    for i in range(len(processed_eigenvectors)):
        print(processed_eigenvectors[i])  # f"第{i+1}組的Eigenvector:",
    print(
        f"FACTOR: {FACTOR}, max_iteration: {max_iteration}, tolerable_error: {tolerable_error}"
    )
    print(f"處理時間: {end_time - start_time} 秒")
    print("\n" + "=" * 50 + "\n")

#     # 構建結果字符串
#     output_lines.append("檔案")
#     output_lines.append(file_path)
#     output_lines.append("FACTOR:")
#     output_lines.append(FACTOR)
#     output_lines.append("max_iteration: ")
#     output_lines.append(max_iteration)
#     output_lines.append("平均值:")
#     output_lines.append(str(mean_values))
#     output_lines.append("共變數矩陣:")
#     for row in cov_matrix:
#         output_lines.append(str(row))
#     output_lines.append("特徵值:")
#     for i, eigenvalue in enumerate(eigenvalues):
#         output_lines.append(f"第{i+1}個特徵值: {eigenvalue}")
#     output_lines.append("特徵向量:")
#     for i, eigenvector in enumerate(eigenvectors):
#         output_lines.append(f"第{i+1}組的特徵向量: {eigenvector}")
#     output_lines.append(f"處理時間: {end_time - start_time} 秒")
#     output_lines.append(
#         "\n"
#         "====================================================================================================="
#         "\n"
#     )

# # 將輸出結果寫入到一個txt文件
# with open("SS_output.txt", "w", encoding="utf-8") as f:
#     for line in output_lines:
#         f.write(str(line) + "\n")
