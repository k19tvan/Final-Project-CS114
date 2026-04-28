# So sánh Generative Learning Algorithm (GLA) và Discriminative Learning Algorithm (DLA)

## 🎯 Mục tiêu đề tài

Dự án nhằm trả lời ba câu hỏi chính:

1. **Khi dữ liệu dồi dào và khan hiếm** – GLA hay DLA sẽ đạt kết quả tốt hơn?
2. **Khi dữ liệu dồi dào và khan hiếm và có nhiễu** – GLA hay DLA sẽ chống chịu nhiễu tốt hơn?
3. **Khi có dữ liệu chưa gán nhãn (unlabeled data)** – GLA hay DLA tận dụng được lợi thế từ dữ liệu chưa gán nhãn tốt hơn?

## 🧪 Phương pháp thực nghiệm

Chúng tôi thực nghiệm trên hai loại bài toán điển hình, sử dụng các cặp thuật toán đối ứng giữa GLA và DLA.

### 1. Phân loại văn bản (Text Classification)

| Loại thuật toán | Mô hình sử dụng |
|----------------|------------------|
| **Discriminative (DLA)** | Logistic Regression |
| **Generative (GLA)** | Naive Bayes |

**Bộ dữ liệu:** AGNews, Sogou, Yelp (Binary và Full), DBPedia, Yahoo Answers.

### 2. Phân loại hình ảnh (Image Classification)

| Loại thuật toán | Mô hình sử dụng |
|----------------|------------------|
| **Discriminative (DLA)** | Logistic Regression |
| **Generative (GLA)** | Gaussian Mixture Models (GMM), Linear Discriminant Analysis (LDA), Gaussian Naive Bayes (GNB) |

**Bộ dữ liệu:** CIFAR-10, MNIST.

## 📂 Cấu trúc thư mục

```
├── data/                           # Dữ liệu thô và đã qua xử lý
│   ├── text/                       # AGNews, Sogou, Yelp, DBPedia, Yahoo...
│   └── image/                      # CIFAR-10, MNIST
│
├── notebooks/                      # Phân tích tổng hợp, so sánh kết quả
│   ├── text_analysis.ipynb         # So sánh GLA vs DLA trên text
│   └── image_analysis.ipynb        # So sánh GLA vs DLA trên ảnh
│
├── src/                            # Thực nghiệm từng phương pháp
│   ├── text/
│   │   ├── naive_bayes.ipynb       # GLA: Naive Bayes
│   │   └── logistic_regression.ipynb # DLA: Logistic Regression
│   └── image/
│       ├── gmm.ipynb               # GLA: GMM
│       ├── lda.ipynb               # GLA: LDA
│       ├── gnb.ipynb               # GLA: Gaussian Naive Bayes
│       └── logistic_regression.ipynb # DLA: Logistic Regression
│
├── results/                        # Lưu kết quả và biểu đồ
│   ├── figures/                    # Biểu đồ so sánh learning curves, noise robustness
│   └── metrics/                    # File csv/json lưu accuracy, F1-score, ...
│
├── README.md
└── requirements.txt
```

## 🛠 Cài đặt và chạy thử nghiệm

1. **Tạo môi trường ảo (khuyến nghị)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

2. **Cài đặt thư viện**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy thực nghiệm**
   - Mở từng notebook trong `src/text/` và `src/image/` để chạy thí nghiệm cho từng mô hình.
   - Kết quả (accuracy, F1, thời gian huấn luyện, …) sẽ tự động được lưu vào `results/metrics/`.
   - Sau đó chạy các notebook trong `notebooks/` để vẽ biểu đồ so sánh và rút ra kết luận.

## 💡 Giả thuyết nghiên cứu (Hypothesis)

- **Generative models (GLA)**  
  - Hoạt động tốt khi **dữ liệu huấn luyện khan hiếm** do học được phân phối đồng thời \(P(x,y)\) và có thể kết hợp tri thức tiên nghiệm.  
  - Dễ dàng tận dụng **dữ liệu chưa gán nhãn** thông qua các kỹ thuật như EM (Expectation‑Maximization).  
  - Nhạy cảm với nhiễu nếu giả định về phân phối (ví dụ: độc lập đặc trưng trong Naive Bayes) bị vi phạm.

- **Discriminative models (DLA)**  
  - Thường đạt độ chính xác cao hơn khi **dữ liệu dồi dào** vì trực tiếp tối ưu ranh giới phân lớp \(P(y|x)\).  
  - Bền vững hơn với nhiễu trong nhiều trường hợp do không phụ thuộc vào giả định phân phối mạnh.  
  - Không khai thác được unlabeled data nếu không có thêm kỹ thuật bán giám sát (self‑training, co‑training,...).

## 📊 Kết quả dự kiến

| Kịch bản | Dự đoán |
|----------|---------|
| Rất ít dữ liệu (cỡ vài trăm mẫu) | Naive Bayes (GLA) > Logistic Regression (DLA) |
| Nhiều dữ liệu (>10.000 mẫu) | Logistic Regression (DLA) ≥ Naive Bayes (GLA) |
| Có nhiễu (label noise, feature noise) | DLA bền vững hơn nếu mức nhiễu vừa phải; GLA suy giảm mạnh nếu giả định phân phối sai |
| Có unlabeled data (semi-supervised) | GLA (GMM, Naive Bayes với EM) cải thiện rõ rệt; DLA cần kỹ thuật bổ sung mới đạt hiệu quả tương tự |

