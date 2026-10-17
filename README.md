# Autoregressive Text Generation: SimpleRNN vs Bidirectional RNN vs LSTM vs GRU

**Course:** Foundations of Generative AI (ICT4403), Assignment 2
**Author:** SAI CHARAN V, 230911368

## 1. Problem statement and objective
Build a word-level autoregressive text-generation system that predicts the next word from the previous words and generates text one word at a time. Four recurrent architectures (SimpleRNN, Bidirectional RNN, LSTM, GRU) are trained under identical conditions and compared on loss, perplexity, accuracy, training time, and the quality of generated text at three sampling temperatures.

## 2. Dataset
- **Name:** *Alice's Adventures in Wonderland* by Lewis Carroll (public domain)
- **Source:** https://www.gutenberg.org/ebooks/11
- **Direct text file used:** https://www.gutenberg.org/cache/epub/11/pg11.txt
- **Characteristics:** about 27k word tokens after cleaning; narrative prose with dialogue, short sentences and a recurring cast (Alice, the Queen, the King, the Mock Turtle, the Cat).
- **Preprocessing:** Gutenberg header and footer removed, lowercased, characters other than letters and `. , ? ! '` removed, punctuation split into separate tokens, vocabulary limited to the 3000 most frequent tokens (the rest mapped to `<unk>`).
- **Input-target sequences:** a sliding window of 10 words predicts the 11th word. The data was shuffled and split 90% training and 10% validation.

## 3. Models
All models share the same pipeline: Embedding(3000, 100) -> recurrent layer (128 units) -> Dropout(0.2) -> Dense(3000, softmax). Adam (lr 1e-3), sparse categorical cross-entropy, 15 epochs, batch size 128, seed 42.

| Model | Recurrent layer |
|---|---|
| SimpleRNN | SimpleRNN(128) |
| BiRNN | Bidirectional(SimpleRNN(128)), so it has more parameters |
| LSTM | LSTM(128) |
| GRU | GRU(128) |

## 4. Execution steps and dependencies
1. Open `FGAI_Assignment2.ipynb` in Google Colab (Runtime -> T4 GPU) or in Jupyter.
2. Run all cells (about 15-20 minutes on a GPU).
3. Dependencies: `tensorflow`, `numpy`, `pandas`, `matplotlib` (all preinstalled on Colab).
4. Outputs are written to `results.csv`, `comparison.png`, `temperature_metrics.csv`, `temperature_repetition.png`, `generated_samples.txt`.

## 5. Results (validation set)

| Model | Params | Train loss | Val loss | Perplexity | Val accuracy | Train time (s) |
|---|---|---|---|---|---|---|
| SimpleRNN | 716,312 | 3.217 | 5.037 | 153.96 | 18.0% | 27.2 |
| BiRNN | 1,129,624 | 2.640 | 5.372 | 215.35 | 16.8% | 24.9 |
| LSTM | 804,248 | 4.158 | 4.966 | 143.40 | 17.6% | 32.6 |
| **GRU** | 775,320 | 3.340 | **4.834** | **125.67** | **20.6%** | 27.1 |

Plots: `comparison.png` (training loss, validation loss, perplexity).

## 6. Model comparison and evaluation
- **GRU** has the lowest validation loss and perplexity and the highest accuracy, at a training time similar to the others.
- **BiRNN** is the weakest. It has the most parameters, the lowest training loss (2.64) but the highest validation loss (5.37), a large gap that indicates overfitting. Looking at both directions does not help here because at generation time only the past words are available.
- **LSTM** is the slowest to train. Its training loss (4.16) is higher than its validation loss (4.97) by a smaller margin than the other models, which suggests it was still learning slowly after 15 epochs.
- **SimpleRNN** sits between BiRNN and LSTM on perplexity.
- Accuracy of 17-21% is expected for next-word prediction over a 3000-word vocabulary on a small corpus, because many continuations are plausible.
- Limitation: each model was trained once with one random seed, so small differences (for example SimpleRNN vs LSTM) should not be over-interpreted.

## 7. Sampling-temperature analysis
Three seeds and temperatures 0.5, 1.0 and 1.5 were used for every model (50 words each). Metrics are averaged over the three seeds (`temperature_metrics.csv`).

| Model | T | distinct-1 | distinct-2 | repeated trigrams |
|---|---|---|---|---|
| SimpleRNN | 0.5 / 1.0 / 1.5 | 0.633 / 0.813 / 0.867 | 0.905 / 0.993 / 1.000 | 0.028 / 0 / 0 |
| BiRNN | 0.5 / 1.0 / 1.5 | 0.627 / 0.747 / 0.867 | 0.878 / 0.973 / 0.993 | 0.056 / 0.014 / 0 |
| LSTM | 0.5 / 1.0 / 1.5 | 0.580 / 0.800 / 0.867 | 0.830 / 0.986 / 1.000 | 0.062 / 0 / 0 |
| GRU | 0.5 / 1.0 / 1.5 | 0.587 / 0.720 / 0.847 | 0.823 / 0.980 / 1.000 | 0.069 / 0 / 0 |

**Observations (GRU samples, seed "alice was beginning to get very tired of sitting by"):**
- **T = 0.5:** the text uses frequent words and mostly sensible phrases from the book ("the mock turtle said", "said the king"), but falls into loops such as "and she went on, and she went on". Coherence is highest locally, diversity is lowest, and repetition is highest.
- **T = 1.0:** vocabulary is richer and the loops disappear, but sentences drift ("the unjust archbishop of the house, and he looked about"). This is the best balance between diversity and readability.
- **T = 1.5:** diversity is near maximum and repetition is zero, but the output is close to word salad with rare words and scattered punctuation ("an newspapers here ! it s the brave will find ...").

Diversity and repetition metrics reward randomness, so they cannot judge coherence alone. Perplexity and reading the samples are needed together.

## 8. Conclusion
**GRU is the most effective model for this dataset.** It achieved the best perplexity (125.7) and accuracy (20.6%) at a comparable training cost, and generated the most book-like phrases at moderate temperature. A temperature between 0.5 and 1.0 gives the best trade-off: low enough to keep sentences readable, high enough to avoid repetitive loops. Temperature 1.5 is too random for coherent text.

**Possible improvements:** a larger corpus, stacked recurrent layers, longer context windows, early stopping, and repeated runs with different seeds.
