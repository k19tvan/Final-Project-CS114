# Machine Learning Classification Pipeline & Robustness Analysis

Dự án này triển khai và so sánh hiệu năng của hai mô hình phân loại chính là **LDA (Linear Discriminant Analysis)** và **SVM (Support Vector Machine)** trên các bộ dữ liệu hình ảnh kinh điển (MNIST, Fashion-MNIST, CIFAR-10). Dự án tập trung vào việc đánh giá độ bền (robustness) của mô hình dưới tác động của nhiễu và khả năng học bán giám sát (Semi-supervised Learning).

---

## 1. Cấu trúc Dự án
```text
.
├── main.py                 # Điểm khởi đầu chương trình (CLI)
├── src/                    # Mã nguồn chính
│   ├── data_loader.py      # Tải và tiền xử lý dữ liệu (Keras/Manual)
│   ├── models.py           # Định nghĩa LDA, SVM wrappers
│   ├── lda_em.py           # Cài đặt LDA-EM và mã nguồn gốc LDA
│   ├── utils.py            # Trích xuất HOG, gây nhiễu, tiện ích vẽ đồ thị
│   ├── experiment1.py      # Baseline & Dimensionality Reduction (PCA)
│   ├── experiment3.py      # Semi-Supervised Learning (Self-Training, EM)
│   └── experiment4.py      # Test-Time Augmentation (TTA) & Reliability
├── results/                # Lưu trữ kết quả (CSV, biểu đồ)
└── readme.md               # Hướng dẫn và thiết lập thí nghiệm
```

---

## 2. Tiền xử lý Dữ liệu & Đặc trưng
Mọi thí nghiệm đều sử dụng đặc trưng **HOG (Histogram of Oriented Gradients)**:
- **Kích thước Cell**: 
  - MNIST / Fashion-MNIST: `(7, 7)` -> `324` features.
  - CIFAR-10: `(8, 8)` -> `324` features.
- **Thông số khác**: `orientations=9`, `cells_per_block=(2, 2)`.
- **Chuẩn hóa**: 
  - SVM luôn đi kèm `StandardScaler` (Z-score normalization).
  - LDA/GLA được huấn luyện trực tiếp hoặc có thể kèm scaling tùy thí nghiệm.

---

## 3. Thiết lập chi tiết các Thí nghiệm

### Thí nghiệm 1: Baseline và Giảm chiều PCA
- **Mục tiêu**: Đánh giá tác động của kích thước tập dữ liệu và số chiều PCA lên độ chính xác.
- **Tham số**:
  - **Subset Sizes**: Từ 20 mẫu đến toàn bộ tập train (60,000 cho MNIST).
  - **PCA Dimensions**: `[2, 5, 10, 15, 20, 50, 100, 150, 200, 250, 300, 324]`.
  - (Đối với CIFAR-10, giới hạn PCA tối đa là 144).
- **Mô hình**: Shrinkage LDA và Linear SVM.

### Thí nghiệm 2: Độ bền với Nhiễu (Noise Robustness)
- **Mục tiêu**: Đánh giá khả năng chịu lỗi của mô hình khi dữ liệu bị "bẩn".
- **Hai kịch bản**:
  - **Subset 500**: Chạy 5 lần (lấy trung bình & độ lệch chuẩn).
  - **Full Dataset**: Chạy 1 lần trên toàn bộ dữ liệu.
- **Loại nhiễu**:
  - **Label Noise**: Xác suất đảo nhãn ngẫu nhiên $p \in [0.0, 0.5]$.
  - **Feature Noise**: Nhiễu Gaussian vào đặc trưng HOG với $\sigma \in [0.0, 1.0]$.

### Thí nghiệm 3: Semi-Supervised Learning (SSL)
- **Mục tiêu**: Tận dụng dữ liệu không nhãn (unlabeled data) khi dữ liệu có nhãn khan hiếm.
- **Tỉ lệ nhãn (Labeled Ratio)**: `[1%, 5%, 10%, 20%, 50%]`.
- **Phương pháp so sánh**:
  - **LDA Self-Training**: Gán nhãn giả (pseudo-label) với threshold tin cậy 0.9.
  - **SVM Self-Training**: Sử dụng `CalibratedClassifierCV`.
  - **LDA-EM**: Tối ưu hóa likelihood thông qua bước E và bước M xen kẽ.

### Thí nghiệm 4: Test-Time Augmentation (TTA) & Reliability
- **Mục tiêu**: Đánh giá hiệu quả của việc tổng hợp kết quả từ nhiều bản sao biến đổi của ảnh test.
- **Cấu hình TTA**: 
  - Số lượng bản sao: $N=10$.
  - **Level 1 (Nhẹ)**: Xoay $\pm 5^\circ$, Dịch $\pm 5\%$, Zoom $\pm 5\%$.
  - **Level 2 (Vừa)**: Xoay $\pm 15^\circ$, Dịch $\pm 10\%$, Zoom $\pm 10\%$.
  - **Level 3 (Mạnh)**: Xoay $\pm 30^\circ$, Dịch $\pm 20\%$, Zoom $\pm 20\%$.
- **Mô hình**: So sánh LDA (Discriminative) và GLA (Generative LDA).
- **Chỉ số**: TTA Accuracy, Consistency Score (độ nhất quán), ECE (Expected Calibration Error).

---

## 4. Cách chạy chương trình

Dự án yêu cầu các thư viện: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scikit-image`, `tensorflow`.

1. **Chạy thí nghiệm cụ thể**:
   ```bash
   python main.py --exp 1  # Baseline & PCA
   python main.py --exp 2  # Noise Robustness
   python main.py --exp 3  # Semi-Supervised Learning
   python main.py --exp 4  # Test-Time Augmentation
   ```

2. **Chạy tất cả**:
   ```bash
   python main.py --all
   ```

---

## 5. Kết quả và Biểu đồ
- Kết quả lưu tại `results/experiment[X]/[dataset_name]/`.
- **Đồ thị**: 
  - Thí nghiệm 1: Accuracy vs PCA Dimensions.
  - Thí nghiệm 2: Robustness curves (Label & Feature noise).
  - Thí nghiệm 3: Accuracy vs Labeled Ratio và Convergence curves.
  - Thí nghiệm 4: TTA Accuracy vs Intensity Levels và Consistency Analysis.

---
**Ghi chú kỹ thuật**: Mô hình LDA sử dụng thuật toán **Shrinkage** (Ledoit-Wolf) để khắc phục hiện tượng ma trận hiệp phương sai bị suy biến khi số lượng mẫu ít hơn số lượng đặc trưng.
