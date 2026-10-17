# Autoregressive Text Generation: SimpleRNN vs Bidirectional RNN vs LSTM vs GRU

**Course:** Foundations of Generative AI (ICT4403), Assignment 2
**Author:** SAI CHARAN V, 230911368

## 1. Problem statement and objective
Build a word-level autoregressive text-generation system that predicts the next word from the previous words and generates text one word at a time. Four recurrent architectures (SimpleRNN, Bidirectional RNN, LSTM, GRU) are trained under identical conditions and compared on loss, perplexity, accuracy and training time, and on the quality of the text they generate (coherence, diversity, contextual consistency, repetition) at three sampling temperatures.

## 2. Dataset
- **Name:** *Alice's Adventures in Wonderland* by Lewis Carroll (public domain)
- **Source:** https://www.gutenberg.org/ebooks/11
- **Direct text file used:** https://www.gutenberg.org/cache/epub/11/pg11.txt
- **Size after cleaning:** 31,508 tokens (words and punctuation marks), 2,579 unique tokens.
- **Characteristics:** narrative prose with a lot of dialogue, short sentences and a recurring cast (Alice, the Queen, the King, the Mock Turtle, the Cat, the Hatter). The most frequent tokens are `,` (2,427), `the` (1,651), `.` (999), `and` (874) and `to` (729).
- **Preprocessing:**
  1. Gutenberg header and footer removed.
  2. Lowercased; characters other than letters, whitespace and `. , ? ! '` removed.
  3. Punctuation split into separate tokens.
  4. Words converted to integer ids with the Keras `Tokenizer` (vocabulary limit 3000, with an `<unk>` token). The corpus has only 2,579 unique tokens, so the limit was never reached and 100% of tokens are in the vocabulary.
- **Input-target sequences:** a sliding window of 10 tokens is the input and the next token is the target. This gives 31,498 sequences, shuffled and split into 28,348 for training and 3,150 for validation.
- **Known limitation:** the source text uses curly apostrophes, which the cleaning step replaced with spaces, so contractions appear as separate tokens in the generated text (for example "it s", "i m", "don t").

## 3. Models
All models share the same pipeline: Embedding(3000, 100) -> recurrent layer (128 units) -> Dropout(0.2) -> Dense(3000, softmax). Adam (learning rate 1e-3), sparse categorical cross-entropy, 15 epochs, batch size 128, random seed 42.

| Model | Recurrent layer |
|---|---|
| SimpleRNN | SimpleRNN(128) |
| BiRNN | Bidirectional(SimpleRNN(128)), so it has more parameters |
| LSTM | LSTM(128) |
| GRU | GRU(128) |

## 4. Execution steps and dependencies
1. Open `FGAI_Assignment2.ipynb` in Google Colab (Runtime -> Change runtime type -> T4 GPU) or in Jupyter.
2. Run all cells (about 15-20 minutes on a GPU).
3. Dependencies: `tensorflow`, `numpy`, `pandas`, `matplotlib` (all preinstalled on Colab). The dataset is downloaded automatically from Project Gutenberg.
4. Outputs are written to `results.csv`, `comparison.png`, `temperature_metrics.csv`, `temperature_repetition.png` and `generated_samples.txt`.

## 5. Results (validation set)

| Model | Params | Train loss | Val loss | Perplexity | Val accuracy | Train time (s) |
|---|---|---|---|---|---|---|
| SimpleRNN | 716,312 | 3.217 | 5.037 | 153.96 | 18.0% | 27.2 |
| BiRNN | 1,129,624 | 2.640 | 5.372 | 215.35 | 16.8% | 24.9 |
| LSTM | 804,248 | 4.158 | 4.966 | 143.40 | 17.6% | 32.6 |
| **GRU** | 775,320 | 3.340 | **4.834** | **125.67** | **20.6%** | 27.1 |

Perplexity = exp(validation loss). Plots: `comparison.png` (training loss, validation loss, perplexity).

## 6. Model comparison and evaluation
- **GRU** has the lowest validation loss and perplexity and the highest accuracy, with a training time similar to the other models.
- **BiRNN** is the weakest. It has the most parameters and the lowest training loss (2.64) but the highest validation loss (5.37). The gap of 2.73 is the largest of the four models and shows that the extra capacity made it memorise the small training text instead of generalising.
- **LSTM** has the highest training loss (4.16) but the smallest train/validation gap (0.81), which suggests it was still learning slowly after 15 epochs. It was also the slowest to train (32.6 s).
- **SimpleRNN** sits between LSTM and BiRNN on perplexity (154.0).
- Accuracy of 17-21% is expected for next-word prediction over a 3000-word vocabulary on a small corpus, because many continuations are plausible at each position.
- Training times differ by only a few seconds, so they should not be used to rank the models.

## 7. Qualitative comparison of generated text
Text was generated from three seed sentences (10 words each), 50 words per sample. Full output: `generated_samples.txt`.

**T = 0.5 (seed: "alice was beginning to get very tired of sitting by")**

| Model | Excerpt | Coherence and consistency | Repetition |
|---|---|---|---|
| SimpleRNN | "the directions , which said the queen s head to her daughter speaking , and the gryphon s grinned in reply ..." | Fluent phrases, but meaning drifts and characters are mixed | Low; some word reuse ("head", "the king") |
| BiRNN | "the hedgehog . the mock turtle the ground and the march hare interrupted the hatter , and the other golden key , and she began out of her hands , and the other golden key ..." | Fluent fragments with a recurring phrase | "the other golden key" twice in one sample |
| LSTM | "the sky . it was a little puppy was a very confused tone , and the mock turtle , and she was not to be ? ..." | Generic filler ("a little", "and she was not") | Repeated "and she was" patterns |
| GRU | "the distant things and the rabbit was questions . and the executioner went on , and she went on , and she went on ..." | Fluent fragments, uses book dialogue ("said the king", "it s the cat") | Clearest loop: "and she went on" three times |

Across all four models the text stays in the vocabulary and cast of the book (queen, king, hatter, mock turtle), so it is in-domain, but it never continues the meaning of the seed. Grammar is locally plausible and sentence-level meaning is poor.

**T = 1.0:** all models produce more varied words but fragmented sentences with rare words and stray tokens, for example GRU: "the unjust archbishop of the house , and he looked about" and SimpleRNN: "the corners man , and alice said all his pieces , upsetting others".

**T = 1.5:** all models produce near word salad, for example LSTM: "again stirring once something like to it throwing matter even the arrived looked on ...".

## 8. Sampling-temperature analysis
Metrics are computed on the generated words and averaged over the three seeds (`temperature_metrics.csv`). distinct-n is the share of unique n-grams (higher means more diverse). Repeated trigrams is the share of trigrams that occur more than once (lower is better).

| Model | T = 0.5 / 1.0 / 1.5 |
|---|---|
| | **distinct-1** |
| SimpleRNN | 0.633 / 0.813 / 0.867 |
| BiRNN | 0.627 / 0.747 / 0.867 |
| LSTM | 0.580 / 0.800 / 0.867 |
| GRU | 0.587 / 0.720 / 0.847 |
| | **distinct-2** |
| SimpleRNN | 0.905 / 0.993 / 1.000 |
| BiRNN | 0.878 / 0.973 / 0.993 |
| LSTM | 0.830 / 0.986 / 1.000 |
| GRU | 0.823 / 0.980 / 1.000 |
| | **Repeated trigrams** |
| SimpleRNN | 0.028 / 0 / 0 |
| BiRNN | 0.056 / 0.014 / 0 |
| LSTM | 0.062 / 0 / 0 |
| GRU | 0.069 / 0 / 0 |

- **Low temperature (0.5):** the most repetitive setting. Repetition is 0.03-0.07 and distinct-2 is lowest. The text is locally fluent but loops. GRU and LSTM, whose probability distributions are sharper, repeat the most; SimpleRNN repeats the least.
- **Medium temperature (1.0):** repetition drops to almost zero and diversity is high, but sentences fragment and rare words appear. This is the best compromise between diversity and repetition in our tests.
- **High temperature (1.5):** diversity is near its maximum and repetition is zero, but the output is close to word salad. Diversity and repetition metrics reward randomness, so they cannot measure coherence; the generated samples have to be read as well.

## 9. Conclusion
**GRU is the most effective model for this dataset** on the main measures of how well the model has learned the language: the lowest validation loss (4.83), the lowest perplexity (125.7) and the highest next-word accuracy (20.6%), at a training cost comparable to the other models. BiRNN is the weakest and overfits most.

For text generation, no model produced globally coherent text; all four give locally fluent but semantically drifting output on this small corpus. The differences in generated text are small and subjective. GRU produced the clearest repetition loop at T = 0.5, while SimpleRNN repeated least, so the generation-quality ranking is not clear-cut. GRU is chosen because its better perplexity and accuracy are the measurable results, and its repetition problem disappears at T = 1.0. A temperature between 0.5 and 1.0 is the best trade-off (intermediate values such as 0.7 were not tested); T = 1.5 is too random.

## 10. Limitations and possible improvements
- Each model was trained once with one random seed, so small differences (for example SimpleRNN vs LSTM) may not be significant.
- The reported training loss is the average over the last epoch with dropout active, while validation loss is measured without dropout, so the train/validation gap slightly overstates overfitting.
- Text metrics use only 3 seeds and one 50-word sample each, and coherence was judged by reading the samples.
- Possible improvements: a larger corpus, handling curly apostrophes, stacked recurrent layers, longer context windows, early stopping, and repeated runs with different seeds.
