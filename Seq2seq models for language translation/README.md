# Seq2seq models for language translation

## Project Overview
This project aims to build and compare machine translation models using Seq2Seq and Transformer architectures. With a focus on translating English to German, the project evaluates different model configurations and attention mechanisms to improve the accuracy and quality of the translations.

## Models Implemented
- **Seq2Seq with RNN/LSTM**: Classic sequence-to-sequence framework using RNN and LSTM units.
- **Seq2Seq with Attention**: Enhancement of the Seq2Seq model by incorporating attention mechanisms.
- **Transformer**: A model relying solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## Libraries
- Python 3.x
- PyTorch
- NumPy

## Results
The project demonstrates the effectiveness of attention mechanisms and the Transformer model in machine translation tasks. Here are some key outcomes:

### Seq2Seq Results – Default Configuration
- **RNN**: Training Perplexity: 70.1586, Validation Perplexity: 73.3413
- **LSTM**: Training Perplexity: 25.7036, Validation Perplexity: 28.9241
- **RNN-with-Attention**: Training Perplexity: 43.6865, Validation Perplexity: 45.1074
- **LSTM-with-Attention**: Training Perplexity: 25.5030, Validation Perplexity: 29.9503

Seq2Seq models with LSTM units showed improved performance over their RNN counterparts. The attention mechanism provided a further boost in the model's ability to focus on relevant parts of the input sequence, leading to lower perplexity scores.

### Transformer Results
- **Encoder Only**: Training Perplexity: 9.5438, Validation Perplexity: 30.3664
- **Full Transformer**: Training Perplexity: 5.1387, Validation Perplexity: 7.4976
- **Best Model (Full Transformer)**: Training Perplexity: 4.0353, Validation Perplexity: 7.1808

The Full Transformer model, with its self-attention and parallelization capabilities, outperformed all other models. It achieved the lowest perplexity scores, indicating a high level of confidence and accuracy in translation.

## Conclusion
The implementation of attention mechanisms has significantly impacted the performance of machine translation models. The Transformer model, with its innovative architecture, has set new standards for the task, providing fast, accurate, and reliable translations.

## Acknowledgments
This work was conducted as part of the [CS 7643 Deep Learning course](https://omscs.gatech.edu/cs-7643-deep-learning) at Georgia Tech. All model implementations and experiments were designed as part of the course's project assignments.
