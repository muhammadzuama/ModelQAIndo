# 🇮🇩 Indonesian Question Answering System using XLM-RoBERTa

Sistem *Question Answering (QA)* ini dibangun menggunakan model `FacebookAI/xlm-roberta-base` yang dilatih dengan data berbahasa Indonesia. Model mampu menjawab pertanyaan berdasarkan konteks yang diberikan secara otomatis.

## 🔗 Akses Cepat

* 💻 **[Google Colab Notebook](https://colab.research.google.com/drive/18r7-yYJNzpSkTUXvnacbwku1g4Lw23_i?usp=sharing)** — Untuk menjalankan pelatihan dan inferensi langsung.
* 🤗 **[Model di Hugging Face Hub](https://huggingface.co/mzuama/my_indo_2_model)** — Unduh atau gunakan model QA yang sudah dilatih.

## 🚀 Fitur

* Preprocessing data `SPAN` untuk ekstraksi jawaban dalam konteks.
* Tokenisasi dengan `XLM-RoBERTa` multilingual tokenizer.
* Training menggunakan `Trainer` dari Hugging Face `transformers`.
* Evaluasi menggunakan skor F1 berbasis token overlap dan `squad` metric.
* Fungsi inferensi pertanyaan berbasis konteks secara langsung.
* Simpan dan muat model dari Google Drive.
* Evaluasi manual dan otomatis atas pertanyaan-pertanyaan berbasis topik tertentu (misalnya sejarah BJ Habibie, AI, dan RRT).

## 🧠 Model

* Base model: `FacebookAI/xlm-roberta-base`
* Tugas: Question Answering (Extractive QA)
* Task type: SQuAD-style span prediction
* Bahasa: Indonesia
* Evaluasi: Custom Span F1 dan `evaluate.load("squad")`

## 🛠️ Cara Menjalankan

### 1. Clone Repository & Install Library

```bash
!git clone https://github.com/username/indo-qa-xlmroberta.git
!pip install transformers datasets evaluate accelerate
```

### 2. Mount Google Drive (Jika menggunakan Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Load Dataset

```python
import pandas as pd
train_data = pd.read_json('/content/drive/MyDrive/RisetQA/df_train.json')
val_data = pd.read_json('/content/drive/MyDrive/RisetQA/df_val.json')
```

### 4. Preprocessing

```python
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_data[train_data['category'] == 'SPAN'].drop(columns=['span_start', 'span_end', 'category']))
```

### 5. Tokenisasi dan Training

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = AutoModelForQuestionAnswering.from_pretrained("FacebookAI/xlm-roberta-base")

# Lanjutkan dengan training menggunakan Hugging Face Trainer
```

### 6. Evaluasi

```python
from evaluate import load
squad_metric = load("squad")

results = squad_metric.compute(predictions=formatted_predictions, references=references)
print(results)
```

### 7. Inferensi

```python
def answer_question(question, context, model, tokenizer):
    # tokenisasi dan prediksi span jawaban
    return answer

answer = answer_question("Siapa presiden pertama Indonesia?", context, model, tokenizer)
print(answer)
```

### Load Model dari Drive

```python
model_path = "/content/drive/MyDrive/my_indo_QA_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

## 📊 Hasil Evaluasi

| **Metric**      | **Value**         |
| --------------- | ----------------- |
| **Loss**        | 1.3813            |
| **Exact Match** | 43.73%            |
| **F1 Score**    | 64.92%            |
| **Runtime**     | 13.14 s           |
| **Eval Speed**  | 48.55 samples/sec |

## 📌 Catatan

* Model hanya dilatih untuk kategori `SPAN`, bukan `YESNO` atau `UNANSWERABLE`.
* Evaluasi masih berbasis *span overlap* sederhana.
* Pertanyaan dengan konteks yang terlalu panjang (>384 token) akan dipotong.

## 🧑‍💻 Kontributor

* [Muhammad Zuama Al Amin](https://github.com/muhammadzuama) – Pengembang utama
