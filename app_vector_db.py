import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import io

# --- Tải dataset ---
# File dữ liệu sẽ được đọc trực tiếp từ thư mục hiện tại
DATASET_PATH = "2cls_spam_text_cls.csv"

# --- Hàm trợ giúp cho Embedding (được tách ra khỏi hàm chính) ---
def average_pool(last_hidden_states, attention_mask):
    """Tính toán average pooling trên last_hidden_states."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(texts, model, tokenizer, device, batch_size=32, progress_callback=None):
    """Tạo embeddings cho một danh sách các văn bản."""
    embeddings = []
    num_texts = len(texts)
    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        batch_dict = tokenizer(batch_texts_with_prefix, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
        if progress_callback:
            # FIX: Cập nhật giá trị tiến trình để không vượt quá 1.0
            progress_callback(min((i + batch_size) / num_texts, 1.0))
    return np.vstack(embeddings)

# --- Chuẩn bị Mô hình Embedding và Vector hóa dữ liệu một lần duy nhất ---
@st.cache_resource
def setup_vector_database():
    """Setup the vector database and return all necessary components."""
    # Kiểm tra xem file có tồn tại không trước khi đọc
    if not os.path.exists(DATASET_PATH):
        st.error(f"Không tìm thấy file dữ liệu: {DATASET_PATH}")
        return None, None, None, None, None, None
    
    df = pd.read_csv(DATASET_PATH)
    messages = df["Message"].values.tolist()
    labels = df["Category"].values.tolist()

    # Chuẩn bị mô hình Embedding
    MODEL_NAME = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Mã hóa nhãn
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Tạo embeddings cho tất cả tin nhắn
    with st.spinner("Đang tạo vector embeddings, vui lòng đợi một lát..."):
        progress_bar = st.progress(0, text="Generating embeddings...")
        X_embeddings = get_embeddings(
            messages,
            model,
            tokenizer,
            device,
            progress_callback=lambda p: progress_bar.progress(p, text=f"Generating embeddings: {int(p * 100)}%")
        )
        progress_bar.empty()
    
    # Tạo metadata cho mỗi tài liệu
    metadata = [{"index": i, "message": message, "label": label} for i, (message, label) in enumerate(zip(messages, labels))]
    
    # Chia dữ liệu và xây dựng FAISS Index
    TEST_SIZE = 0.1
    SEED = 42
    train_indices, _ = train_test_split(range(len(messages)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train_emb = X_embeddings[train_indices]
    train_metadata = [metadata[i] for i in train_indices]
    
    embedding_dim = X_train_emb.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(X_train_emb.astype("float32"))

    return index, model, tokenizer, device, train_metadata, df

index, model, tokenizer, device, train_metadata, df = setup_vector_database()

# --- Logic phân loại ---
def classify_with_knn(query_text, model, tokenizer, device, index, train_metadata, k=3):
    """Phân loại văn bản bằng cách sử dụng k-lân cận gần nhất với embeddings"""
    query_with_prefix = f"query: {query_text}"
    batch_dict = tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors="pt")
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        query_embedding = query_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(query_embedding, k)
    predictions = []
    neighbor_info = []
    for i in range(k):
        neighbor_idx = indices[0][i]
        neighbor_score = scores[0][i]
        neighbor_label = train_metadata[neighbor_idx]["label"]
        neighbor_message = train_metadata[neighbor_idx]["message"]
        predictions.append(neighbor_label)
        neighbor_info.append({
            "score": float(neighbor_score),
            "label": neighbor_label,
            "message": neighbor_message
        })
    unique_labels, counts = np.unique(predictions, return_counts=True)
    final_prediction = unique_labels[np.argmax(counts)]
    return final_prediction, neighbor_info

# --- Streamlit UI ---
st.title("Ứng dụng Phân loại Tin nhắn Spam (Vector Database)")
st.write("Sử dụng cơ sở dữ liệu vector (FAISS) và mô hình ngôn ngữ để phân loại tin nhắn.")

if index is not None:
    st.success("Mô hình và cơ sở dữ liệu vector đã được tải thành công!")
    st.write(f"Cơ sở dữ liệu chứa **{len(train_metadata)}** tin nhắn đã được vector hóa.")
    
    user_input = st.text_area("Nhập tin nhắn của bạn vào đây:", height=150)
    k_value = st.slider("Số lượng lân cận gần nhất (k)", 1, 10, 3)
    
    if st.button("Dự đoán"):
        if user_input:
            with st.spinner(f'Đang tìm kiếm {k_value} lân cận gần nhất...'):
                prediction, neighbors = classify_with_knn(user_input, model, tokenizer, device, index, train_metadata, k=k_value)
            
            st.subheader("Kết quả dự đoán:")
            if prediction == 'spam':
                st.error(f"Đây là tin nhắn: **{prediction.upper()}**")
            else:
                st.success(f"Đây là tin nhắn: **{prediction.upper()}**")
            
            st.subheader(f"Top {k_value} lân cận gần nhất:")
            for i, neighbor in enumerate(neighbors, 1):
                st.write(f"**{i}. Tin nhắn tương tự ({neighbor['label'].upper()})** | Điểm số: {neighbor['score']:.4f}")
                st.write(f"_{neighbor['message']}_")
        else:
            st.warning("Vui lòng nhập một tin nhắn để dự đoán.")
else:
    st.error("Không thể tải hoặc thiết lập cơ sở dữ liệu vector. Vui lòng kiểm tra lại.")
