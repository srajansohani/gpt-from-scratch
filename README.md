

# 🧠 Building GPT from Scratch: A Comprehensive Journey

This repository documents my journey of **rebuilding a GPT-style language model from scratch**. While initially inspired by the excellent tutorial by Andrej Karpathy, this project has evolved into a much broader exploration of Large Language Model architectures.

In addition to foundational concepts, this repository now includes custom implementations of key transformer components (such as GELU activations, Layer Normalization, and varied attention mechanisms) and goes beyond a simple reproduction.

Along the way, I focus on:
- **Modular Component Implementation**: Building and understanding individual blocks like FeedForward Layers and LayerNorm from the ground up.
- My own **experiments** and architectural variations.
- Deeper **intuitions** into the math and engineering behind modern LLMs.
- Practical **modifications and improvements**.

---

## 🚀 Motivation

Instead of just using pre-trained models, I wanted to truly understand:

- How tokens become embeddings  
- How transformers process sequences  
- How GPT generates text autoregressively  
- What actually happens during training  

This repo is my attempt to **bridge theory → implementation → intuition**.

---

## 📚 What This Project Covers

### Core Implementation
- Tokenization (character / subword level)
- Embedding layer
- Self-attention mechanism
- Multi-head attention
- Transformer blocks
- Layer normalization
- Feedforward networks
- Training loop
- Text generation

---

### 🧪 Extensions (My Additions)

- Custom dataset experiments  
- Hyperparameter tuning  
- Alternative tokenization approaches  
- Training optimizations  
- Attention visualization (planned)  
- Notes on scaling behavior  

---

## 🏗️ Project Structure
