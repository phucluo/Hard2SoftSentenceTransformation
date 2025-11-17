import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# =========================
# 1. Dataset song song
# =========================
love_pairs = [
    {
        "id": 1,
        "hard": "Anh đang bận, đừng nhắn nữa.",
        "soft": "Anh đang hơi bận một chút, em cho anh xin ít thời gian, xong việc anh nhắn lại em liền nhé. 💌"
    },
    {
        "id": 2,
        "hard": "Em phiền quá.",
        "soft": "Chắc do anh hơi mệt nên phản ứng chưa được tốt, nhưng anh biết em quan tâm anh, cảm ơn em nhiều nha. 🫶"
    },
    {
        "id": 3,
        "hard": "Anh không muốn đi với em đâu.",
        "soft": "Hôm nay anh hơi đuối, mình hẹn dịp khác để anh có nhiều năng lượng dành cho em hơn nhé. 💖"
    },
    {
        "id": 4,
        "hard": "Đừng gọi cho anh nữa.",
        "soft": "Anh cần chút không gian để sắp xếp lại mọi thứ, xong anh sẽ chủ động liên lạc với em nhé."
    },
    {
        "id": 5,
        "hard": "Em nói nhiều quá.",
        "soft": "Tự nhiên hôm nay anh hơi mệt nên tiếp thu chậm, mình nói từ từ từng chuyện một được không em? 🥺"
    },
    {
        "id": 6,
        "hard": "Anh không quan tâm mấy chuyện đó.",
        "soft": "Anh chưa hiểu rõ lắm chuyện này, em kể thêm cho anh với để anh hiểu em hơn nha."
    },
    {
        "id": 7,
        "hard": "Thích thì làm, anh chịu.",
        "soft": "Anh hơi bối rối nên chưa biết quyết sao, hay là mình bàn thêm chút để tìm cách tốt nhất cho cả hai ha?"
    },
    {
        "id": 8,
        "hard": "Tuỳ em.",
        "soft": "Anh tin vào lựa chọn của em, nếu em cần ý kiến thêm thì anh luôn sẵn sàng góp ý nè. 😊"
    },
    {
        "id": 9,
        "hard": "Anh mệt em lắm rồi.",
        "soft": "Dạo này anh hơi áp lực nên dễ cáu, nhưng anh vẫn rất trân trọng em. Mình cùng tìm cách để cả hai đỡ mệt hơn nhé."
    },
    {
        "id": 10,
        "hard": "Đừng nhắn nữa, phiền.",
        "soft": "Anh đang bị overload chút, cho anh xin tạm nghỉ điện thoại, tối anh ổn hơn anh nhắn em nha."
    },
    {
        "id": 11,
        "hard": "Anh không muốn nghe em than nữa.",
        "soft": "Anh sợ mình không cho em được lời khuyên tốt nhất lúc này, nhưng anh vẫn luôn ở đây để lắng nghe em, mình nói chậm lại một chút nha."
    },
    {
        "id": 12,
        "hard": "Em ghen vô lý quá.",
        "soft": "Anh hiểu em quan tâm nên mới để ý nhiều vậy, mình cùng nói rõ cho nhau hiểu hơn để em yên tâm hơn nha."
    },
    {
        "id": 13,
        "hard": "Anh chán rồi.",
        "soft": "Anh đang thấy mối quan hệ mình có vài chỗ chưa ổn, mình thử ngồi lại nói chuyện để cải thiện được không em?"
    },
    {
        "id": 14,
        "hard": "Anh không thích kiểu em làm vậy.",
        "soft": "Có vài điều anh hơi chưa quen với cách em làm, mình bàn với nhau xem có cách nào hợp với cả hai hơn không em?"
    },
    {
        "id": 15,
        "hard": "Đừng làm phiền anh nữa.",
        "soft": "Anh đang cần tập trung một chút, em cho anh xin ít thời gian, xong anh quay lại với em nha. 💗"
    },
    {
        "id": 16,
        "hard": "Anh thấy em vô lý lắm.",
        "soft": "Anh hơi khó hiểu với cách em nhìn chuyện này, em giúp anh hiểu góc nhìn của em hơn được không?"
    },
    {
        "id": 17,
        "hard": "Anh không muốn nhắn tin nữa.",
        "soft": "Anh hơi mệt mắt vì điện thoại rồi, mình nói chuyện tiếp sau nhé, anh không muốn trả lời em trong trạng thái không tốt."
    },
    {
        "id": 18,
        "hard": "Anh không có thời gian cho em.",
        "soft": "Dạo này lịch của anh hơi dày, nhưng anh vẫn muốn sắp xếp thời gian cho em, mình cùng tìm khung giờ hợp lý hơn nha."
    },
    {
        "id": 19,
        "hard": "Em suy nghĩ nhiều quá.",
        "soft": "Anh hiểu em lo, nhưng nhiều khi em đang tự làm mình mệt hơn, để anh cùng em gỡ từng chuyện một nha."
    },
    {
        "id": 20,
        "hard": "Chuyện đó không quan trọng.",
        "soft": "Có vẻ anh chưa cảm nhận hết được tầm quan trọng của chuyện này với em, em kể kỹ hơn cho anh hiểu được không?"
    },
    {
        "id": 21,
        "hard": "Anh thấy em trẻ con quá.",
        "soft": "Có vài phản ứng của em làm anh hơi bất ngờ, mình thử tìm cách nói chuyện với nhau chín chắn hơn một xíu nha."
    },
    {
        "id": 22,
        "hard": "Anh không thích bạn em.",
        "soft": "Anh hơi lo khi em chơi với vài người bạn, chắc do anh sợ em bị ảnh hưởng, mình nói kỹ hơn để hiểu nhau hơn nhé."
    },
    {
        "id": 23,
        "hard": "Anh không muốn nói về chuyện đó nữa.",
        "soft": "Anh sợ nếu nói tiếp lúc này sẽ dễ làm em buồn, mình tạm dừng một chút, khi cả hai bình tĩnh hơn mình nói tiếp nha."
    },
    {
        "id": 24,
        "hard": "Kệ em.",
        "soft": "Anh tôn trọng quyết định của em, nếu em muốn nghe thêm góc nhìn của anh thì anh luôn sẵn sàng chia sẻ."
    },
    {
        "id": 25,
        "hard": "Em làm anh khó chịu.",
        "soft": "Có vài điều khiến anh hơi khó chịu, nhưng anh muốn nói với em một cách để mình cùng sửa, chứ không phải trách em."
    },
    {
        "id": 26,
        "hard": "Anh không muốn gặp em.",
        "soft": "Anh đang không ở trạng thái tốt nhất, anh sợ gặp em sẽ làm em buồn, cho anh xin thời gian ổn lại rồi mình gặp nhau nhé."
    },
    {
        "id": 27,
        "hard": "Đừng hỏi nữa.",
        "soft": "Anh hơi mệt khi nghĩ về chuyện này, mình tạm gác lại một chút, tối anh bình tĩnh hơn mình nói tiếp nha."
    },
    {
        "id": 28,
        "hard": "Anh lười nói chuyện với em.",
        "soft": "Hôm nay anh hơi cạn năng lượng, nên nói chuyện không được nhiệt tình như bình thường, em đừng nghĩ là anh không muốn nói với em nha."
    },
    {
        "id": 29,
        "hard": "Anh không thích em làm vậy với người khác.",
        "soft": "Anh hơi khó chịu khi thấy em như vậy với người khác, chắc do anh hơi nhạy cảm, mình cùng bàn xem đâu là ranh giới thoải mái cho cả hai nha."
    },
    {
        "id": 30,
        "hard": "Đừng đăng mấy cái đó nữa.",
        "soft": "Anh hơi lo khi thấy em đăng mấy bài đó, anh sợ người khác hiểu sai về em, mình nói với nhau xem có cách khác để em chia sẻ cảm xúc không nha."
    },
]

love_df = pd.DataFrame(love_pairs)

# =========================
# 2. Load model + precompute embeddings
# =========================
@st.cache_resource
def load_embed_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    return model

@st.cache_resource
def build_index(df):
    model = load_embed_model()
    hard_sentences = df["hard"].tolist()
    hard_embs = model.encode(hard_sentences, convert_to_tensor=True, show_progress_bar=False)
    return hard_sentences, df["soft"].tolist(), hard_embs

def rewrite_loving(user_text: str, top_k: int = 3, min_sim: float = 0.35):
    if not user_text or not user_text.strip():
        return []

    model = load_embed_model()
    hard_sents, soft_sents, hard_embs = build_index(love_df)

    query_emb = model.encode(user_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, hard_embs)[0]

    top_k = min(top_k, len(hard_sents))
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        score_val = float(score.cpu().item())
        if score_val < min_sim:
            continue
        idx = int(idx)
        results.append(
            {
                "hard_template": hard_sents[idx],
                "soft_suggestion": soft_sents[idx],
                "similarity": score_val,
            }
        )
    return results

# =========================
# 3. Streamlit UI
# =========================
st.set_page_config(page_title="Cỗ máy dịch lời khó nói", page_icon="💌")

st.title("💌 Cỗ Máy Dịch Lời Khó Nói Sang Lời Dễ Thương")
st.write(
    """
Nhập một câu hơi **thẳng, khó nói** với người yêu, 
app sẽ gợi ý phiên bản **dịu dàng, dễ thương** hơn mà vẫn giữ ý chính.  

Dưới lớp vỏ lãng mạn là **sentence embeddings + semantic search**.  
"""
)

with st.expander("Xem một vài ví dụ có sẵn trong dataset"):
    st.table(love_df[["hard", "soft"]].head(5).rename(columns={"hard": "Câu khó nói", "soft": "Câu dễ thương"}))

user_input = st.text_area(
    "Nhập câu bạn định nhắn (càng thật càng tốt 😅)",
    height=100,
    placeholder="Ví dụ: Anh đang bận, đừng làm phiền anh nữa."
)

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.slider("Số gợi ý muốn xem", min_value=1, max_value=5, value=3)
with col2:
    min_sim = st.slider("Ngưỡng độ giống (similarity)", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

if st.button("Biến hoá cho mềm hơn ✨"):
    if not user_input.strip():
        st.warning("Nhập gì đó trước đã nha. 🥹")
    else:
        with st.spinner("Đang suy nghĩ câu trả lời dễ thương hơn..."):
            candidates = rewrite_loving(user_input, top_k=top_k, min_sim=min_sim)

        if not candidates:
            st.info("Tớ chưa tìm được câu nào đủ giống trong dataset. Bạn có thể mở rộng dataset thêm sau này.")
        else:
            best = candidates[0]
            st.subheader("💘 Gợi ý phù hợp nhất")
            st.write(best["soft_suggestion"])

            st.caption(f"(Dựa trên template: “{best['hard_template']}”, similarity ≈ {best['similarity']:.2f})")

            if len(candidates) > 1:
                st.markdown("---")
                st.subheader("Các gợi ý khác")
                for i, cand in enumerate(candidates[1:], start=2):
                    with st.container(border=True):
                        st.markdown(f"**Phương án {i}**")
                        st.write(cand["soft_suggestion"])
                        st.caption(
                            f"Template: “{cand['hard_template']}”  ·  similarity ≈ {cand['similarity']:.2f}"
                        )

st.markdown("---")
st.markdown(
    """
**Giải thích kỹ thuật (tóm tắt):**  
- Embed câu của bạn và các câu “khó nói” trong dataset bằng model đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2`.  
- Tìm câu “khó nói” gần nhất (cosine similarity) → lấy bản “dễ thương” tương ứng.  
- Có thể mở rộng dataset để app ngày càng giống… chuyên gia tư vấn tình yêu hơn. 💫
"""
)
