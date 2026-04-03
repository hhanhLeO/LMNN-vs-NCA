from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Tải tập dữ liệu Wine
wine = datasets.load_wine()
X = wine.data
y = wine.target

# 2. Chia tập huấn luyện và kiểm tra (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train[:5])
# ==========================================
# KỊCH BẢN 1: KHÔNG DÙNG SCALER
# ==========================================
# Khởi tạo và huấn luyện mô hình trực tiếp trên dữ liệu gốc
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

# Dự đoán và tính độ chính xác
y_pred_unscaled = knn_unscaled.predict(X_test)
acc_unscaled = accuracy_score(y_test, y_pred_unscaled)


# ==========================================
# KỊCH BẢN 2: CÓ DÙNG SCALER (STANDARDSCALER)
# ==========================================
scaler = StandardScaler()

# Bước quan trọng: Fit và transform trên tập Train
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled[:5])

# CHỈ transform trên tập Test (sử dụng mean và std đã học từ tập Train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình trên dữ liệu đã chuẩn hóa
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

# Dự đoán và tính độ chính xác
y_pred_scaled = knn_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, y_pred_scaled)


# ==========================================
# IN KẾT QUẢ SO SÁNH
# ==========================================
print(f"Độ chính xác KHÔNG có Scaler: {acc_unscaled * 100:.2f}%")
print(f"Độ chính xác CÓ dùng Scaler:   {acc_scaled * 100:.2f}%")