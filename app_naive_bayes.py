import streamlit as st
import string
import nltk
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# --- Tải dataset ---
# File dữ liệu sẽ được đọc trực tiếp từ thư mục hiện tại
DATASET_PATH = "2cls_spam_text_cls.csv"

# --- Cài đặt NLTK data (nếu chưa có) ---
@st.cache_resource
def setup_nltk():
    """
    Setup NLTK resources by downloading them if they are not found.
    Using @st.cache_resource ensures this is only run once per session.
    """
    st.info("Kiểm tra và tải tài nguyên NLTK...")
    
    # Định nghĩa thư mục lưu trữ NLTK trong dự án hiện tại
    nltk_data_path = "nltk_data"
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    # Thêm đường dẫn tùy chỉnh vào NLTK
    nltk.data.path.append(nltk_data_path)

    # Kiểm tra và tải 'stopwords'. Phần này vẫn hoạt động tốt.
    try:
        nltk.data.find('corpora/stopwords', paths=[nltk_data_path])
    except LookupError:
        st.warning("Không tìm thấy tài nguyên 'stopwords'. Đang tải xuống...")
        nltk.download('stopwords', download_dir=nltk_data_path)
        st.success("Tải xuống 'stopwords' thành công.")
    
    # Do lỗi với 'punkt', chúng ta sẽ bỏ qua việc tải xuống tài nguyên này
    # và sử dụng một bộ tokenizer dựa trên biểu thức chính quy thay thế.
    st.success("Tất cả tài nguyên NLTK đã sẵn sàng!")
    return True

if not setup_nltk():
    st.stop()

# --- 1. Tiền xử lý dữ liệu ---
def preprocess_text(text):
    """Thực hiện các bước tiền xử lý văn bản:
    1. Chuyển thành chữ thường
    2. Loại bỏ dấu câu
    3. Mã hóa (tokenize) bằng regex
    4. Loại bỏ stop words
    5. Stemming (đưa từ về dạng gốc)
    """
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Thay thế word_tokenize bằng tokenizer dựa trên biểu thức chính quy để tránh lỗi LookupError.
    tokens = re.findall(r'\b\w+\b', text)
    
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# --- 2. Tạo đặc trưng (Bag-of-Words) ---
def create_dictionary(messages):
    """Tạo từ điển (dictionary) từ tất cả các token trong dữ liệu đã tiền xử lý."""
    word_to_index = {}
    index = 0
    for tokens in messages:
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = index
                index += 1
    return word_to_index

def create_features(tokens, dictionary):
    """Chuyển đổi danh sách token thành vector đặc trưng Bag-of-Words."""
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary[token]] += 1
    return features

# --- Huấn luyện mô hình và tiền xử lý dữ liệu một lần duy nhất ---
@st.cache_resource
def train_model():
    """Train the Naive Bayes model and return the necessary components."""
    # Kiểm tra sự tồn tại của file dữ liệu
    if not os.path.exists(DATASET_PATH):
        st.error(f"Không tìm thấy file dữ liệu: {DATASET_PATH}")
        return None, None, None, None
            
    df = pd.read_csv(DATASET_PATH)
    messages = df["Message"].values.tolist()
    labels = df["Category"].values.tolist()
    
    # Tiền xử lý tin nhắn
    preprocessed_messages = [preprocess_text(message) for message in messages]
    
    # Tạo từ điển
    dictionary = create_dictionary(preprocessed_messages)
    
    # Tạo features (Bag-of-Words)
    X = np.array([create_features(tokens, dictionary) for tokens in preprocessed_messages])
    
    # Mã hóa nhãn
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Chia bộ dữ liệu
    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.125, shuffle=True, random_state=0)
    
    # Huấn luyện mô hình
    model = GaussianNB()
    model.fit(X_train_val, y_train_val)
    
    return model, dictionary, le, df

model, dictionary, label_encoder, df = train_model()

# --- Hàm dự đoán cho tin nhắn mới ---
def predict(text, model, dictionary, label_encoder):
    """Dự đoán nhãn cho một tin nhắn mới."""
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = label_encoder.inverse_transform(prediction)[0]
    return prediction_cls

# --- Streamlit UI ---
st.title("Ứng dụng Phân loại Tin nhắn Spam (Naive Bayes)")
st.write("Sử dụng thuật toán Naive Bayes để phân loại tin nhắn là 'spam' hoặc 'ham' (bình thường).")

if model:
    st.success("Mô hình đã được tải và huấn luyện thành công!")
    st.write(f"Bộ dữ liệu có **{df.shape[0]}** tin nhắn.")
    
    user_input = st.text_area("Nhập tin nhắn của bạn vào đây:", height=150)
    
    if st.button("Dự đoán"):
        if user_input:
            with st.spinner('Đang dự đoán...'):
                prediction = predict(user_input, model, dictionary, label_encoder)
                
            st.subheader("Kết quả dự đoán:")
            if prediction == 'spam':
                st.error(f"Đây là tin nhắn: **{prediction.upper()}**")
            else:
                st.success(f"Đây là tin nhắn: **{prediction.upper()}**")
        else:
            st.warning("Vui lòng nhập một tin nhắn để dự đoán.")
else:
    st.error("Không thể tải hoặc huấn luyện mô hình. Vui lòng kiểm tra lại đường dẫn dữ liệu.")
