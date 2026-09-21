# Story Generation with an LSTM

Give the model a few words, and it keeps writing.

```
Seed:   "first citizen we are"
Output: "first citizen we are a king and what i know not not not done you
         that we will not the crown and a king i fear you have all the
         house of a world that i am not the point of a king charity"
```

The model reads Shakespeare, learns which word tends to follow which, and then
writes new lines one word at a time.

---

## The dataset

**Tiny Shakespeare** — 1.1 MB of Shakespeare's plays, public domain.

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The notebook downloads it automatically, so there is nothing to set up.

| | |
|---|---|
| Lines | 32,777 |
| Words | 202,619 |
| Different words | 12,848 |

---

## How to run it

1. Open `LSTM_Story_Generator_Simple.ipynb` in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**

It finishes in about two minutes.

---

## Results

| | Loss | Accuracy |
|---|---|---|
| Training | 6.3464 | 0.0835 |
| Validation | 6.3917 | 0.0840 |
| **Test** | 6.6809 | **0.0774** |

**What these numbers mean.** The model picks the correct next word about **8%**
of the time. That sounds low until you consider it is choosing from **12,848
possible words** — random guessing would be right 0.008% of the time, so the
model is roughly **995× better than chance**.

The more useful number is the small gap between training (0.0835) and test
(0.0774). The model does nearly as well on text it has never seen as on text it
trained on, which means it learned real patterns instead of memorising the
script.

---

## Why we built it this way

### Why a simple flow

We started from a straightforward, well-understood approach: take a few words,
predict the next one, repeat. No clever tricks. A simple pipeline that we could
fully understand and verify was worth more than a complicated one we could not
check in the time available.

### Why we had to change one thing

The usual version of this code turns each answer into a long list of zeros with
a single one in it — one slot for every word in the vocabulary. With 169,842
training examples and 12,848 words, that list would have needed **8.7 GB of
memory** and crashed the session before training started.

The fix was to keep each answer as a plain number (word #4,318 instead of a list
of 12,848 zeros) and let the training function look it up. Same result, a few
megabytes instead of 8.7 GB, and slightly less code.

**This is the one change that made the approach work on a full-sized dataset.**

### Why the model picks words randomly instead of always the best one

Always choosing the single most likely word sounds correct, but it gets stuck:

> and the king and the king and the king and the king...

Because the choice is always the same, the model loops forever. Instead it picks
from the likely options with some randomness. A `temperature` setting controls
how adventurous it is — low is safe and repetitive, high is varied and messier.
Both are shown in the notebook.

### Why only 5 epochs

An epoch is one pass through all the training text. We stopped at 5 for two
reasons:

1. **It stopped helping.** Validation results were best at epoch 4 and got
   slightly worse afterwards. Training longer would have produced a *worse*
   model, not a better one. The notebook automatically keeps the best version.
2. **Time.** The whole task had a hard deadline, so a two-minute training run
   that we could repeat and verify beat a long run we would only get one shot
   at.

### Why we measure three scores instead of one

The text is split three ways:

- **Training** (about 80%) — what the model learns from
- **Validation** (10%) — checked after each pass to catch it going wrong
- **Test** (10%) — held back completely, never used during training

The test portion matters because the validation portion helps *decide* which
version of the model to keep, so it is no longer a neutral judge. The test set
had no say in anything, so it is the honest score.

---

## The model

```
Embedding          turns each word into a list of 100 numbers
Bidirectional LSTM reads the sentence forwards and backwards (150 units)
Dense (softmax)    gives every one of the 12,848 words a probability
```

About 5.4 million adjustable values, trained with the Adam optimiser.

---

## Honest limitations

- **It repeats itself.** Common words like "king" and "the" show up often. With
  202,619 words and a 12,848-word vocabulary, the average word appears only
  about 16 times — not many examples to learn from.
- **It does not tell a story.** It predicts one word at a time and has no sense
  of plot, characters, or an ending. It captures the *style* of Shakespeare, not
  the structure of a story.
- **No punctuation.** The task asked for punctuation to be removed, so the
  output has none.
- **Rare words never get learned.** A word appearing once or twice does not give
  the model enough to work with.

**The biggest improvement would be more text, not a bigger model.** The full
works of Shakespeare are about five times this size, and that would help more
than any change to the architecture.

---

## Files

| File | What it is |
|---|---|
| `LSTM_Story_Generator_Simple.ipynb` | The whole project — run this |
| `input.txt` | The Shakespeare text |
| `.gitignore` | Keeps the 62 MB trained model out of the repo |

The trained model file (`word_lstm.keras`) is not included because it is 62 MB
and the notebook recreates it in about two minutes.
