import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from linearRegression import LinearRegression


def tai_du_lieu(duong_dan: str):
    df = pd.read_csv(duong_dan)

    # Loại bỏ các cột không cần thiết
    # Thêm 'ad_id' nếu có trong file dummies.csv và không phải là feature
    # Nếu 'dummies.csv' được tạo từ dataExplore.ipynb, nó sẽ không có 'ad_id' nữa.
    # errors='ignore' sẽ bỏ qua nếu cột không tồn tại.
    columns_to_drop = ['Price', 'log_price']
    if 'ad_id' in df.columns:  # Kiểm tra xem cột ad_id có tồn tại không
        columns_to_drop.append('ad_id')

    X = df.drop(columns=columns_to_drop, errors='ignore')

    # Chuẩn hóa toàn bộ X
    scaler = StandardScaler()
    # Đảm bảo X chỉ chứa các cột số trước khi chuẩn hóa
    X_numeric = X.select_dtypes(include=np.number)
    X_scaled_values = scaler.fit_transform(X_numeric)

    # Tạo lại DataFrame X_scaled với các cột đã chuẩn hóa và giữ lại các cột không phải số nếu có
    X_scaled_df = pd.DataFrame(X_scaled_values, columns=X_numeric.columns, index=X.index)


    # Lấy log_price làm y
    y_log = df['log_price'].values
    y = df['Price'].values  # Price gốc để so sánh cuối cùng

    # Trả về DataFrame X chuẩn hóa và tên cột
    return X_scaled_df, y_log, y, X_numeric.columns.tolist()


def chia_train_test(X, y, ti_le_test=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    test_len = int(len(X) * ti_le_test)
    test_idx, train_idx = idx[:test_len], idx[test_len:]
    # Đảm bảo X và y được индексируются bằng iloc nếu là DataFrame/Series
    X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
    X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test


def main():

    file_csv = os.path.join(os.path.dirname(__file__), 'dataset.csv')

    print('Đang tải và xử lý dữ liệu...')
    X, y_log, y_original_price, ten_cot = tai_du_lieu(file_csv)
    print(f'Dữ liệu sau xử lý: {X.shape[0]} dòng, {X.shape[1]} biến.')

    # Sử dụng y_log cho huấn luyện
    X_train, X_test, y_train_log, y_test_log = chia_train_test(X, y_log)

    print('Huấn luyện mô hình Gradient Descent (trên log_price)...')
    model = LinearRegression(learning_rate=0.2, n_iterations=4000)
    model.fit(X_train.values, y_train_log)  # Sử dụng .values để truyền numpy array vào LinearRegression

    # Lưu weights và bias nếu cần
    model.weights.tofile('weights.dat')
    np.array([model.bias]).tofile('bias.dat') # Bias là một scalar, cần chuyển thành array để lưu

    y_pred_log = model.predict(X_test.values)  # Sử dụng .values

    # Đánh giá trên thang log
    mse_log = np.mean((y_test_log - y_pred_log) ** 2)
    r2_log = 1 - np.sum((y_test_log - y_pred_log) ** 2) / np.sum((y_test_log - np.mean(y_test_log)) ** 2)

    # Quy đổi về giá trị gốc để đánh giá và trực quan hóa
    y_test_price = np.exp(y_test_log)
    y_pred_price = np.exp(y_pred_log)

    mse_price = np.mean((y_test_price - y_pred_price) ** 2)
    r2_price = 1 - np.sum((y_test_price - y_pred_price) ** 2) / np.sum((y_test_price - np.mean(y_test_price)) ** 2)

    print('\nKẾT QUẢ')
    print(f'- Intercept (Bias): {model.bias:.4f}')
    if model.weights is not None and len(ten_cot) == len(model.weights):
        print('- 10 hệ số quan trọng nhất:')
        sorted_indices = np.argsort(np.abs(model.weights))[::-1]
        for i, idx in enumerate(sorted_indices[:10]):
            if i < len(ten_cot):  # Đảm bảo idx nằm trong phạm vi của ten_cot
                print(f'  {ten_cot[idx]:<45} {model.weights[idx]:>10.4f}')
            else:
                print(f'  Index {idx} out of bounds for feature names.')
    else:
        print("Không thể hiển thị hệ số quan trọng do lỗi kích thước hoặc weights chưa được huấn luyện.")

    print('\n— Đánh giá trên thang log (log_price) —')
    print(f'  • MSE (log_price): {mse_log:.4f}')
    print(f'  • R²  (log_price): {r2_log:.4f}')

    print('\n— Đánh giá quy đổi về **Giá** —')
    print(f'  • MSE (price)    : {mse_price:,.0f}')
    print(f'  • R²  (price)    : {r2_price:.4f}')

    # --- Tối ưu biểu đồ Thực tế vs Dự đoán (Giá) ---
    plt.figure(figsize=(8, 8))  # Tăng kích thước để dễ nhìn hơn
    plt.scatter(y_test_price, y_pred_price, alpha=0.5, s=35, edgecolors='k',
                linewidths=0.5)  # s: kích thước điểm, thêm viền

    # Xác định giới hạn cho đường tham chiếu để bao phủ toàn bộ dữ liệu
    min_val = min(y_test_price.min(), y_pred_price.min())
    max_val = max(y_test_price.max(), y_pred_price.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)  # Đường màu đỏ, nét đứt, đậm hơn

    plt.xlabel('Giá thực tế', fontsize=12)
    plt.ylabel('Giá dự đoán', fontsize=12)
    plt.title('Thực tế vs Dự đoán (Giá)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)  # Thêm lưới dạng chấm, mờ

    # Đảm bảo trục bắt đầu từ 0 nếu tất cả giá trị là dương
    current_xlim = plt.xlim()
    current_ylim = plt.ylim()
    plt.xlim(left=max(0, current_xlim[0]))
    plt.ylim(bottom=max(0, current_ylim[0]))

    # Thêm thông số R² và MSE vào biểu đồ
    plt.text(0.05, 0.95, f'R²: {r2_price:.4f}\nMSE: {mse_price:,.0f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()  # Tự động điều chỉnh bố cục
    plt.show()

    # --- Biểu đồ hội tụ Gradient Descent ---
    plt.figure(figsize=(8, 5))  # Điều chỉnh kích thước
    plt.plot(model.cost_history)
    plt.xlabel('Vòng lặp (Iterations)', fontsize=12)
    plt.ylabel('MSE (trên log_price)', fontsize=12)
    plt.title('Biểu đồ hội tụ của hàm mất mát (Gradient Descent)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()