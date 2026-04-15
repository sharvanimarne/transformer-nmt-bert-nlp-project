# 🚀 Transformer from scratch (PyTorch) + BERT fine-tuning | NLP + Deep Learning Project

This project is a complete implementation of the paper **"Attention Is All You Need" (Vaswani et al., 2017)** along with a modern NLP pipeline using **BERT fine-tuning**.

It covers:
- Natural Language Processing (NLP)
- Attention Mechanisms & Transformers

---

## 📌 Project Overview

This project consists of two major parts:

### 1. Transformer from Scratch (Neural Machine Translation EN → FR)
- Built entire Transformer architecture manually
- Includes:
  - Positional Encoding
  - Scaled Dot-Product Attention
  - Multi-Head Attention (8 heads)
  - Encoder-Decoder architecture
  - Masked + Cross Attention

- Training techniques:
  - Noam Learning Rate Scheduler
  - Label Smoothing Loss

---

### 2. BERT Fine-Tuning (Sentiment Analysis)
- Pretrained BERT model used
- Fine-tuned for classification task
- Includes:
  - Tokenization
  - AdamW optimizer
  - Linear warmup scheduler

---

## ⚙️ Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- NumPy, Matplotlib
- Google Colab (GPU)

---

## 📂 Project Structure
transformer_bert_nlp.py # Complete implementation
README.md

---

## ▶️ How to Run

1. Open **Google Colab**
2. Upload the `.py` file
3. Enable GPU: Runtime → Change Runtime Type → T4 GPU
4. Run all cells sequentially

⏱ Runtime: ~25–35 minutes

---

## Results

### Transformer (NMT)
- BLEU-1 Score: 100
- BLEU-2 Score: 100

### BERT (Sentiment Analysis)
- Accuracy: 100%
- Precision: 100
- Recall: 100
- F1 Score: 100

---

## 📈 Visualizations

- Training Loss Curve
<img width="1207" height="332" alt="Screenshot (1202)" src="https://github.com/user-attachments/assets/bf7edf77-ca47-46be-867c-ea3e22de07bb" />
- Positional Encoding Heatmap
<img width="740" height="318" alt="image" src="https://github.com/user-attachments/assets/d49a429b-0804-45d3-bf45-0b0a4012ec2a" />
- Attention Heatmaps (All 8 Heads)
<img width="1087" height="463" alt="image" src="https://github.com/user-attachments/assets/1394caca-8f7f-4de2-b079-c681931bafd7" />
- Confusion Matrix
<img width="412" height="366" alt="image" src="https://github.com/user-attachments/assets/f32a04c4-cc80-4dda-820e-e4bf4b50e43a" />

---

## 🧠 Key Learnings

- Deep understanding of Transformer architecture
- Implementation of attention mechanisms from scratch
- Real-world NLP pipeline using BERT
- Model training optimization techniques

---

## 🔮 Future Improvements

- Beam Search decoding instead of Greedy
- Larger datasets for better translation
- Fine-tuning GPT/T5 for comparison
- Deploy as web application

---

## 📚 References

1. Vaswani et al., 2017 — Attention Is All You Need  
2. Devlin et al., 2018 — BERT  
3. HuggingFace Documentation  

---

## 👩‍💻 Author

**Sharvani Marne**  
Third Year IT Engineering Student  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
