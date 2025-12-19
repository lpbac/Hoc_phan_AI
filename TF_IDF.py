import pandas as pd
import numpy as np
import re
import os
from underthesea import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# --- CẤU HÌNH ---
INPUT_FILE = "data - data.csv"
TOP_K_SIMILAR = 3 

def load_and_clean_data(path):
    """
    Phiên bản V2: Tăng cường khả năng nhận diện tên cột (chữ thường/hoa).
    """
    print(f"🔄 Đang đọc và xử lý dữ liệu từ {path}...")
    
    if not os.path.exists(path):
        print(f"❌ Lỗi: Không tìm thấy file '{path}'. Hãy chắc chắn file nằm cùng thư mục code.")
        exit()

    try:
        df = pd.read_csv(path)
        print(f"   Các cột gốc: {df.columns.tolist()}")

        # 1. Tự động tìm cột chứa Văn bản (Thêm 'comment' viết thường)
        col_text = None
        # Ưu tiên các tên cột phổ biến
        candidates_text = ['comment', 'Comment', 'content', 'text', 'review', 'binh_luan']
        for candidate in candidates_text:
            if candidate in df.columns:
                col_text = candidate
                break
        
        # 2. Tự động tìm cột chứa Điểm số/Nhãn
        col_label = None
        # Ưu tiên 'label' trước (vì thường đã được xử lý), sau đó mới đến 'rate'/'rating'
        candidates_label = ['label', 'rate', 'Rating', 'star', 'stars', 'point']
        for candidate in candidates_label:
            if candidate in df.columns:
                col_label = candidate
                break

        if not col_text or not col_label:
            print(f"❌ Vẫn không tìm thấy cột phù hợp.")
            print(f"   Hãy kiểm tra kỹ file CSV. Code đang tìm các tên: {candidates_text} và {candidates_label}")
            exit()

        print(f"   -> Chọn cột văn bản: '{col_text}'")
        print(f"   -> Chọn cột nhãn: '{col_label}'")

        # 3. Chuẩn hóa tên cột
        df = df.rename(columns={col_text: 'text', col_label: 'label_original'})

        # 4. Xử lý Logic Nhãn
        # Nếu cột là 'label' (thường là 0/1 hoặc pos/neg) -> giữ nguyên hoặc map lại
        # Nếu cột là 'rate' (thường là điểm số) -> quy đổi
        
        # Kiểm tra xem dữ liệu trong cột label là số hay chữ
        unique_vals = df['label_original'].unique()
        print(f"   -> Các giá trị mẫu trong cột nhãn: {unique_vals[:5]}")

        if pd.api.types.is_numeric_dtype(df['label_original']):
            # Nếu giá trị lớn nhất > 1 (ví dụ thang điểm 5, 10) -> Cần quy đổi
            if df['label_original'].max() > 1:
                print("   -> Đang quy đổi điểm số (Rate) sang nhãn 0/1...")
                # Logic: Rate >= 6 (thang 10) hoặc >= 4 (thang 5) là Tích cực
                # Tự động đoán thang điểm dựa trên max
                threshold = 6.0 if df['label_original'].max() > 5 else 3.5 
                df['label'] = df['label_original'].apply(lambda x: 1 if x >= threshold else 0)
            else:
                # Nếu chỉ có 0 và 1 -> Giữ nguyên
                print("   -> Dữ liệu đã là nhãn 0/1.")
                df['label'] = df['label_original']
        else:
            # Nếu là chuỗi (ví dụ 'positive', 'negative')
            print("   -> Đang chuyển đổi nhãn dạng chữ sang số...")
            df['label'] = df['label_original'].apply(lambda x: 1 if str(x).lower() in ['pos', 'positive', 'tich cuc'] else 0)

        # Loại bỏ dòng trống
        df = df.dropna(subset=['text'])
        
        texts = df["text"].astype(str).reset_index(drop=True)
        labels = df["label"].reset_index(drop=True)
        
        print(f"✅ Đã xử lý xong! Tổng số mẫu: {len(df)}")
        return texts, labels

    except Exception as e:
        print(f"❌ Lỗi xử lý dữ liệu: {e}")
        # In ra chi tiết lỗi để debug
        import traceback
        traceback.print_exc() 
        exit()

def preprocess_text(text):
    """Hàm tiền xử lý văn bản tiếng Việt."""
    text = str(text).lower() # Chuyển thành string đề phòng lỗi
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        text = word_tokenize(text, format="text")
    except:
        pass # Bỏ qua nếu lỗi tách từ
    return text

def main():
    # 1. Load & Clean Data
    texts, labels = load_and_clean_data(INPUT_FILE)

    # 2. Vectorization
    print("🔄 Đang vector hóa văn bản (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        max_features=5000,
        ngram_range=(1, 2) 
    )
    
    X_full = vectorizer.fit_transform(texts)

    # 3. Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 4. Huấn luyện
    print("🔄 Đang huấn luyện mô hình Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 5. Đánh giá
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*30)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # 6. Dự đoán & Similarity
    print("\n" + "="*30)
    print("THỬ NGHIỆM DỰ ĐOÁN & TÌM KIẾM TƯƠNG ĐỒNG")
    print("="*30)

    while True:
        raw_text = input("\n✍️  Nhập văn bản (gõ 'exit' để thoát): ")
        if raw_text.strip().lower() == "exit":
            break
        if not raw_text.strip(): continue

        text_vec = vectorizer.transform([raw_text])
        
        # Dự đoán
        prediction = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec).max()
        label_text = "Tích cực" if prediction == 1 else "Tiêu cực"
        
        print(f"\n🔮 Kết quả: {label_text.upper()} (Độ tin cậy: {proba:.2%})")

        # Tìm kiếm tương đồng
        print(f"🔎 Top {TOP_K_SIMILAR} review tương tự nhất:")
        cosine_sim_scores = cosine_similarity(text_vec, X_full).flatten()
        related_indices = cosine_sim_scores.argsort()[-TOP_K_SIMILAR:][::-1]

        print("-" * 50)
        for i, idx in enumerate(related_indices, 1):
            score = cosine_sim_scores[idx]
            if score > 0.1:
                original_text = texts.iloc[idx]
                original_label = "Tích cực" if labels.iloc[idx] == 1 else "Tiêu cực"
                # Hiển thị tối đa 100 ký tự
                display_text = (original_text[:100] + '...') if len(original_text) > 100 else original_text
                
                print(f"#{i} (Giống: {score:.2%}) [{original_label}]: \"{display_text}\"")
            else:
                print(f"#{i}: Không tìm thấy bài viết tương tự.")
        print("-" * 50)

if __name__ == "__main__":
    main()