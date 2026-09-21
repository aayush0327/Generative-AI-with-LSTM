# Story Generator with an LSTM

Give the model a few words to start with, and it keeps writing the rest.

```
Seed:   "first citizen before we proceed"
Output: "first citizen before we proceed the officers o' the people in the
         refell'd of pates mistaking washed iniquity credit countrymen
         testimony highness' bond manners wolf argues gripe undo afflict..."
```

That is real output from the finished model, not a tidied-up version. The first
half holds together; after that it drifts. The rest of this page explains how it
works, why we built it this way, and how good it actually is.

---

## The flow, in plain words

The whole project is one notebook with five steps. Nothing more.

**1. Read the text.**
We open `stories.txt` and lowercase everything. Lowercasing means "The" and
"the" count as the same word, so the model has fewer things to learn.

**2. Give every word a number.**
A model cannot read letters, only numbers. So `the` becomes 1, `and` becomes 2,
and so on. This is called tokenizing.

**3. Turn each line into practice questions.**
From the line *"once upon a time"* we make:

| Question (what the model sees) | Answer (what it should say) |
|---|---|
| once | upon |
| once upon | a |
| once upon a | time |

Every line in the file gives us a handful of these. That is how a plain text
file becomes training data — no labelling by hand, no extra work.

**4. Make all questions the same length.**
Some questions are 3 words, some are 15. The model needs one fixed size, so we
pad the short ones with zeros at the front.

**5. Train, then write.**
The model learns from all those questions. After that, we hand it a starting
phrase, it predicts the next word, we stick that word on the end, and we ask
again. Repeat 50 times and you have a paragraph.

---

## The dataset

| | |
|---|---|
| File | `stories.txt` (Shakespeare, public domain) |
| Lines | 40,000 |
| Words | 202,651 |
| Different words | 12,633 |
| Practice questions built from it | 171,312 |

---

## Results

We ran the same notebook three times, stopping at **20, 50 and 70 passes** over
the text. Here is where it landed each time:

| Passes (epochs) | Accuracy | Loss | Training time |
|---|---|---|---|
| 20 | 46.0% | 2.63 | ~8 minutes |
| 50 | 73.2% | 1.25 | ~19 minutes |
| **70** | **77.2%** | **1.01** | **~27 minutes** |

On a free Colab T4 GPU, one pass takes about 23 seconds.

**What the accuracy number means.** The model picks the correct next word 77%
of the time. It is choosing from 12,633 possible words, so random guessing would
score 0.008% — the model is thousands of times better than chance.

**What it does not mean.** This notebook trains on all the text and scores
itself on the same text. So 77% mostly says *"it has learned this file very
well"*, not *"it writes good English"*. Those are different things, and the
output below shows the gap.

### More passes raised the score but not the quality

This was the most useful thing we learned. Accuracy climbed steadily — 46% to
73% to 77% — but the writing did not get three-quarters better. Here is the real
70-pass output:

```
WHAT IS THE CITY BUT THE PEOPLE
> what is the city but the people with thee ever with to him i charge knell
  you puissant mowbray grey you as now you as you the york's wife ebb south

MY LORD I DO BESEECH YOU
> my lord i do beseech you hear me make it stand i ' good servants as the
  duke caused burn finds mistaking senator treading idly insinuate peer

O GENTLE ROMEO
> o gentle romeo this was not even on a loud ere relent and reigns now eat
  bias destroy you as now now body the execute trumpet incaged now purge
```

The first few words after the seed are usually fine. Then it slides into rare
words strung together. Training longer made it better at reciting the file it
had already seen, and no better at continuing a sentence it had not.

**The honest takeaway: with this much text, more training passes are not the
missing ingredient. More text is.**

---

## Why we chose this flow

### Because it is small enough to understand completely

There are 15 short cells in the notebook. Every single one can be read,
explained and checked in a minute. A pipeline we fully trust is worth more than
a clever one we cannot verify.

### Because we were short on time

This is the honest reason behind most of the decisions here. We had a hard
deadline, so the rule we followed was:

> Pick the version that trains fast enough to run again if something breaks.

A setup that takes two hours gives you one attempt. A setup that takes twenty
minutes gives you several — which is exactly why we could try 20, 50 and 70
passes and compare them instead of guessing.

### The one thing we had to change

The obvious version of this code turns every answer into a long row of zeros
with a single 1 in it — one slot for every word in the vocabulary. On this file
that is:

> 171,312 answers x 12,633 words x 4 bytes = **8.66 GB**

Colab has about 12 GB of RAM, so **the session crashed before training even
started.** This actually happened to us.

The fix is to keep each answer as a plain word number (word #4,318 instead of a
row of 12,633 zeros) and let the loss function do the lookup. Exactly the same
maths, exactly the same result:

| | Memory |
|---|---|
| One row of zeros per answer | 8.66 GB — crashes |
| Plain word number per answer | **0.69 MB** — fine |

In the code that is a two-word change: `categorical_crossentropy` becomes
`sparse_categorical_crossentropy`, and the `to_categorical(...)` line is
deleted. **This is the single change that makes the simple flow work on a
full-sized file.**

We also removed the per-operation device logging the first version had switched
on. It printed a line for every operation TensorFlow ran, which filled the
notebook with millions of lines of noise and ate memory on its own.

---

## Why each piece is there

| Piece | Why we used it |
|---|---|
| **Keras / TensorFlow** | Free, runs on Google Colab's free GPU, and the whole model is 5 lines. No setup on our own machines. |
| **Google Colab** | Free GPU. 23 seconds per pass there; on a laptop CPU the same pass takes several minutes. |
| **Embedding layer** | Lets the model learn that "king" and "queen" are related. Without it, every word is just an unrelated number. |
| **Bidirectional LSTM** | LSTM remembers earlier words in the line, so it does not forget the subject halfway through. "Bidirectional" means it reads the context both ways. |
| **Dense + softmax** | Gives every one of the 12,633 words a score. |
| **Adam optimiser** | The safe default. Works well without us tuning the learning rate — one less thing to spend time on. |
| **Batches of 128** | The default of 32 wastes the GPU. Bigger batches cut a pass down to 23 seconds. |
| **A 16-word window** | The model looks back at most 16 words. Longer context would cost memory and time for very little gain on lines this short. |
| **Plain text file as input** | No database, no cleaning scripts, no API. Drop in a `.txt` and the notebook handles the rest. |

### The model

```
Embedding          turns each word into 200 numbers        2,526,600 values
Bidirectional LSTM reads the line forwards and backwards     641,600 values
Dense (softmax)    scores all 12,633 words                 5,065,833 values
```

**8,234,033 adjustable values in total (31 MB).**

---

## How to run it

1. Open `Using_LSTM_as_story_generator.ipynb` in Google Colab
2. Upload `stories.txt` (the notebook reads it from `/content/stories.txt`)
3. **Runtime → Change runtime type → T4 GPU**
4. **Runtime → Run all**

To change how long it trains, edit one number:

```python
history = model.fit(predictors, label, epochs=70, batch_size=128, verbose=1)
```

To change what it writes about, edit the seed:

```python
input_text = "first citizen before we proceed"
print(generate_story(input_text, 50, model, max_length, temperature=0.8))
```

Use words that appear in `stories.txt` — the model only knows the words it was
trained on, and anything else is simply ignored.

`temperature` controls how adventurous it is: `0` always takes the most likely
word (and tends to loop — *"and the king and the king..."*), higher numbers are
more varied and messier. `0.8` is what we used.

---

## What this model does not do

Being honest about the limits is part of the point of keeping it simple:

- **It has no plot.** It predicts one word at a time. It does not know how a
  story starts or ends — only what tends to follow what.
- **It drifts after a few words.** Shown above. The further it gets from the
  seed, the less the words hold together.
- **It has never been tested on unseen text.** There is no validation or test
  split in this notebook, so 77% is a score on text it has already studied. A
  split would give a lower but more honest number.
- **It only knows words it has seen.** With 202,651 words spread over a
  12,633-word vocabulary, the average word appears about 16 times — not many
  examples to learn from, and rare words never get learned at all.
- **No punctuation or capital letters.** Everything is lowercased, and
  punctuation is dropped by the tokenizer.

**If we had more time, in order:** more text (the full works of Shakespeare are
about five times this file), then a proper train/test split so we can measure it
honestly, then a longer context window.

---

## Files

| File | What it is |
|---|---|
| `Using_LSTM_as_story_generator.ipynb` | The whole project — run this |
| `stories.txt` | The text the model learns from |
| `word_lstm.keras` | The trained model (not in git — 62 MB, the notebook rebuilds it) |
| `.gitignore` | Keeps the big model file out of the repo |
