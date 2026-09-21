# Story Generator with an LSTM

Give the model a few words to start with, and it keeps writing the rest.

```
Seed:   "In the hustle and bustle of ipoti"
Output: "in the hustle and bustle of ipoti the taj mahal taught anaya the
         importance of love devotion and the power of architectural beauty
         of the human spirit they stood on the centuries the friends
         discovered ancient inscriptions and hidden chambers..."
```

That is the whole idea. The model reads a lot of text, learns which word usually
comes after which, and then writes new text one word at a time.

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
Some questions are 3 words, some are 40. The model needs one fixed size, so we
pad the short ones with zeros at the front.

**5. Train, then write.**
The model learns from all those questions. After that, we hand it a starting
phrase, it predicts the next word, we stick that word on the end, and we ask
again. Repeat 50 times and you have a paragraph.

---

## Why we chose this flow

### Because it is small enough to understand completely

There are about 12 short cells in the notebook. Every single one can be read,
explained and checked in a minute. A pipeline we fully trust is worth more than
a clever one we cannot verify.

### Because we were short on time

This is the honest reason behind most of the decisions here. We had a hard
deadline, so the rule we followed was:

> Pick the version that trains fast enough to run again if something breaks.

A setup that takes two hours gives you one attempt. A setup that takes a few
minutes gives you ten. When time is tight, the number of attempts matters more
than the sophistication of any single one.

### Because it actually worked

We ran this exact flow on a small collection of short stories and the model
reached about **96% accuracy** on the training text, with output that reads
like real sentences (the sample at the top of this page). It finished in roughly
20 seconds per pass on a free Colab GPU.

That was the moment we stopped looking for a better approach. The simple thing
was already producing the result we needed.

---

## Why each piece is there

| Piece | Why we used it |
|---|---|
| **Keras / TensorFlow** | Free, runs on Google Colab's free GPU, and the whole model is 5 lines. No setup on our own machines. |
| **Google Colab** | Free GPU. Training on a laptop CPU would have taken hours instead of minutes. |
| **Embedding layer** | Lets the model learn that "king" and "queen" are related. Without it, every word is just an unrelated number. |
| **Bidirectional LSTM** | LSTM remembers earlier words in the sentence, so it does not forget the subject halfway through. "Bidirectional" means it reads the context both ways, which improved the sentences noticeably for almost no extra cost. |
| **Dense + softmax** | Gives every word in the vocabulary a score, and we take the highest one. |
| **Adam optimiser** | The safe default. It works well without us having to tune the learning rate — one less thing to spend time on. |
| **Plain text file as input** | No database, no cleaning scripts, no API. Drop in a `.txt` and the notebook handles the rest. |

---

## How to run it

1. Open `Using LSTM as story generator.ipynb` in Google Colab
2. Upload `stories.txt` (the notebook reads it from `/content/stories.txt`)
3. **Runtime → Change runtime type → T4 GPU**
4. **Runtime → Run all**

Then edit the last cell to change the starting phrase:

```python
input_text = "In the hustle and bustle of"
print(generate_story(input_text, 50, model, max_length))
```

---

## A note about the numbers

The results saved inside the notebook come from a run on a **small collection of
short stories** (about 2,600 different words). That is where the 96% accuracy
and the sample output above came from.

The `stories.txt` in this folder is a **much larger text** — 40,000 lines and
about 12,600 different words. If you run the notebook on this file, expect:

- Longer training time (many more practice questions to work through)
- A lower accuracy number, because the model is now choosing between 12,600
  words instead of 2,600

Lower accuracy on the bigger file is not a bug. It is a harder test. Both runs
use exactly the same code.

---

## What this model does not do

Being honest about the limits is part of the point of keeping it simple:

- **It has no plot.** It predicts one word at a time. It does not know how a
  story starts or ends — it only knows what tends to follow what.
- **It repeats itself.** We always take the single most likely next word, so
  the model can fall into loops like *"and the king and the king..."*. Picking
  randomly among the top few words would fix this, and is the first thing we
  would add with more time.
- **It only knows words it has seen.** A word that appeared twice in the file
  will never be used well.
- **It learns the file closely.** With 100 training passes on a small file, a
  lot of what it produces is close to the original text. More text would help
  here far more than a bigger model would.

**If we had more time, in order:** more text, then random word picking instead
of always-the-best, then a proper train/test split to measure it honestly.

---

## Files

| File | What it is |
|---|---|
| `Using LSTM as story generator.ipynb` | The whole project — run this |
| `stories.txt` | The text the model learns from |
| `word_lstm.keras` | The trained model (not in git — 62 MB, the notebook rebuilds it) |
| `.gitignore` | Keeps the big model file out of the repo |
