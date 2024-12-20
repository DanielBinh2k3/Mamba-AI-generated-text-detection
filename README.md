# Enhancing AI Text Detection with MambaFormer and Adversarial Learning

This repository contains the code and results for our research on enhancing AI text detection using the MambaFormer model and adversarial learning techniques.  Our work focuses on improving the accuracy and robustness of AI-generated text detection models, addressing the growing need to identify synthetically generated content.

## Motivation

The proliferation of generative AI tools like ChatGPT presents both opportunities and challenges. While these tools can be valuable learning aids, they also pose a risk to academic integrity and the development of critical thinking skills.  This research aims to develop robust methods for detecting AI-generated text, ensuring that students engage in authentic writing and learning.

## Literature Review

Our work builds upon existing research in AI text detection, including:

* **Zero-shot methods like DetectGPT [2]:**  Leveraging probability curvature for detection, but susceptible to distribution shifts and paraphrasing attacks.
* **Supervised methods like RADAR [3]:** Employing adversarial learning with paraphrasers and detectors in a feedback loop, demonstrating improved robustness.
* **State Space Models (SSMs) [4] and Mamba [5]:**  Introducing dynamic parameters and selective scanning for enhanced context representation and efficient sequence processing.

## Problem Statement

Given a dataset of human-written and AI-generated sentences, our goal is to accurately classify the origin of each text segment.  We aim to enhance model accuracy and resilience against diverse writing styles and cross-domain content by integrating advanced techniques like adversarial learning and model fusion.

## Main Contributions

* **MambaFormer Model Construction:**  Inspired by "An Empirical Study of Mamba-based Language Models" [6], we leverage Mamba's recurrent architecture for efficient long sequence processing and eliminate the need for positional encoding through a hybrid model with FlashAttention.
* **Adversarial Learning Methodology:**  Building on the RADAR [3] framework, we employ dynamic adversarial training with an ensemble of paraphrasers to enhance the detector's robustness against evolving AI-generated text variations.
* **Continuous Adaptation:**  The system continuously adapts to maintain its effectiveness against emerging patterns in AI-generated text.

## Implementation and Results

* **Training Dataset:** 200k rows of mixed LLM-generated writing data.
* **Models:** Deberta-base, MambaFormer.
* **Training Arguments:**
    * Learning rate: 1e-6
    * Optimizer: AdamW
    * Learning rate scheduler: Cosine
    * Batch size: 32
    * Gradient accumulation: 4
* **Metrics:** F1 score, AUROC.
* **Validation Datasets:** DetectGPT's [3] XSum, Writing, and PubMed datasets, along with AI-generated text from GPT-4.
* **Adversarial Training:** Wikipedia Intro dataset, T5-large, BART, and Pegasus paraphrasers.  Validation performed on DetectGPT datasets and Mistral-7B paraphrased text (1-4 times).

Results demonstrate the potential of MambaFormer in text classification, outperforming existing methods.

## Conclusion

This research contributes to the field of AI-generated text detection by exploring the effectiveness of MambaFormer and adversarial learning.  Our findings highlight the potential of these techniques for improving detection accuracy and robustness.

## Future Work

* **Model Distillation:**  Creating lighter and more efficient models for faster deployment.
* **Exploration of New Language Models:**  Experimenting with architectures like xLSTM for enhanced text understanding.
* **Ensemble Methods:**  Combining multiple detection models, including watermark detectors, to leverage their individual strengths.


## References

1. Abdali, S., et al. (2024). Decoding the AI Pen: Techniques and Challenges in Detecting AI-Generated Text. arXiv:2403.05750.
2. Mitchell, E., et al. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. ICML.
3. Hu, X., Chen, P.-Y., & Ho, T.-Y. (2023). RADAR: Robust AI-text detection via adversarial learning. NeurIPS 36.
4. Gu, A., et al. (2021). Combining recurrent, convolutional, and continuous-time models with linear state space layers. NeurIPS 34.
5. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
6. Krishna, K., et al. (2024). Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. NeurIPS 36.
