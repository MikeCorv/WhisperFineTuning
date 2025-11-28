# 🇮🇹 Italian ASR Fine-Tuning: Whisper v3 Turbo + LoRA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Training%20Complete-success)

This repository contains an end-to-end pipeline for fine-tuning OpenAI's **Whisper Large v3 Turbo** on the Italian **FLEURS** dataset (Available on HuggingFace [here](https://huggingface.co/datasets/google/fleurs)

The project demonstrates how to train a State-of-the-Art (SOTA) Speech-to-Text model on consumer hardware (Google Colab T4 GPU) using **QLoRA** (Quantized Low-Rank Adaptation).

## 🏆 Results



| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Base Model** | `openai/whisper-large-v3-turbo` | Released late 2024 |
| **Dataset** | Google FLEURS (Italian) | ~3,000+ audio samples |
| **Final WER** | **3.95%** | [Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer)|
| **Training Time** | ~1 Hour | Single NVIDIA T4 (Available on Colab for free) |

---

## 🧠 Technical Approach

Training a 1.5 Billion parameter model usually requires massive industrial GPUs. To make this work on a free 16GB GPU, **Parameter-Efficient Fine-Tuning (PEFT)** was necessary. More details here: https://huggingface.co/blog/peft.

### 1. 4-Bit Quantization
The base model was loaded using **NF4 (NormalFloat 4)** quantization. This compresses the model weights from 16-bit precision down to 4-bit, reducing VRAM usage by ~4x without significant loss in intelligence.

### 2. LoRA
Instead of retraining the entire model (which remains frozen), tiny trainable matrices ('Adapters') are injected into the Attention layers.
* **Frozen Parameters:** ~800 Million (99.2%)
* **Trainable Parameters:** ~15 Million (0.8%)

### 3. Data Engineering
* **Filtering:** Removed audio < 1s (silence) and > 30s (OOM risk).
* **Resampling:** Converted all inputs to 16kHz.
* **Firewall Collator:** Implemented a custom Data Collator to strip `input_ids` and prevent Trainer conflicts.

---

## 📂 Project Structure

| Notebook | Description |
| :--- | :--- |
| `GoogleFleursAudio.ipynb` | Downloads FLEURS, cleans text, resamples audio, and filters outliers. Saves processed Arrow files to Drive. |
| `WHISPERV3TURBOFINETUNING.ipynb` | Loads the model in 4-bit, attaches LoRA adapters, runs the training loop, and saves the final adapters. |
| `testing.ipynb` | *(Planned)* Loads the saved adapters from Drive for fast transcription of new audio files. |

---

## 🚀 Usage

(NOTE: you may require other libraries, check the notebooks.)

### Installation
```bash
pip install transformers datasets peft bitsandbytes accelerate librosa
