# 🛡️ RAC-FraudNet: Continual Learning for Spam Detection

**RAC-FraudNet** (Robust Adversarial Continual Fraud Network) is an advanced NLP framework designed to combat evolving fraudulent messages. Unlike static filters, this system utilizes **Prototype-Based DistilBERT** and **Elastic Weight Consolidation (EWC)** to learn from new scam patterns in real-time without forgetting previous knowledge.

## 🚀 Key Features

* **Distance-Based Classification**: Uses learnable prototypes (centroids) instead of standard linear heads to improve feature clustering.
* **Stability-Plasticity Balance**: Implements a dual-memory system (Experience Replay + EWC) to prevent catastrophic forgetting.
* **Self-Tuning Bandit Policy**: An automated Upper Confidence Bound (UCB1) controller that optimizes learning rates and replay ratios during online adaptation.
* **Adversarial Resilience**: Tested against LLM-generated (Groq-powered) adversarial "whitening" scams.

---

## 📂 Project Structure

* `dl-project (11).ipynb`: The core research notebook containing the training pipeline, Bandit controller, and EWC implementation.
* `app.py`: A local Streamlit interface for real-time message analysis.
* `rac_fraudnet_best.pt`: The optimized model weights (Note: Hosted externally/locally).
* `requirements.txt`: Necessary dependencies (`torch`, `transformers`, `streamlit`).

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/RAC-FraudNet.git
cd RAC-FraudNet

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the Local Detector

Ensure your `rac_fraudnet_best.pt` file is in the root directory or update the path in `app.py`.

```bash
streamlit run app.py

```

---

## 🧪 Research Methodology

1. **Phase 1 (Offline)**: Base training on standard SMS and Smishing datasets to establish foundational "Ham" vs "Spam" boundaries.
2. **Phase 2 (Online)**: Continuous micro-updates using the Bandit Policy to adapt to specific "whitening" product fraud and Ghar Soap scam variations.
3. **Evaluation**: Measured via **Plasticity** (learning new tasks) and **Stability** (retaining old tasks).

---

## 📊 Results Summary

| Metric | Value |
| --- | --- |
| **Model Architecture** | Prototype-Based DistilBERT |
| **Optimal Threshold** | ~0.5 (Dynamic) |
| **Stabilization** | EWC (λ=30,000) |

---

## 🤝 Acknowledgments

Special thanks to my mentor, **Anjali T**, and the research team at **Amrita Vishwa Vidyapeetham** for their guidance on this project.
