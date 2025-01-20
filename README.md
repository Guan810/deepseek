---
base_model: deepseek-ai/DeepSeek-R1
language:
- en
library_name: transformers
tags:
- deepseek
- unsloth
- transformers
---

# We are working on DeepSeek-R1 GGUFs. =)

---

## ***See [our collection](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5) for versions of Deepseek R1 including GGUF and original formats.***


# Finetune LLMs 2-5x faster with 70% less memory via Unsloth!
We have a free Google Colab Tesla T4 notebook for Llama 3.1 (8B) here: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord%20button.png" width="200"/>](https://discord.gg/unsloth)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

# unsloth/DeepSeek-R1
For more details on the model, please go to Deepseek's original [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)

## ✨ Finetune for Free

All notebooks are **beginner friendly**! Add your dataset, click "Run All", and you'll get a 2x faster finetuned model which can be exported to GGUF, vLLM or uploaded to Hugging Face.

| Unsloth supports          |    Free Notebooks                                                                                           | Performance | Memory use |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **Llama-3.2 (3B)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)               | 2.4x faster | 58% less |
| **Llama-3.2 (11B vision)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)               | 2x faster | 60% less |
| **Qwen2 VL (7B)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb)               | 1.8x faster | 60% less |
| **Qwen2.5 (7B)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb)               | 2x faster | 60% less |
| **Llama-3.1 (8B)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2.4x faster | 58% less |
| **Phi-3.5 (mini)** | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3.5_Mini-Conversational.ipynb)               | 2x faster | 50% less |
| **Gemma 2 (9B)**      | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma2_(9B)-Alpaca.ipynb)               | 2.4x faster | 58% less |
| **Mistral (7B)**    | [▶️ Start on Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb)               | 2.2x faster | 62% less |

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="200"/>](https://docs.unsloth.ai)

- This [Llama 3.2 conversational notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb) is useful for ShareGPT ChatML / Vicuna templates.
- This [text completion notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb) is for raw text. This [DPO notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) replicates Zephyr.
- \* Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster.

## Special Thanks
A huge thank you to the DeepSeek team for creating and releasing these models.