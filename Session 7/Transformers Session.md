---
title: Transformers Session

---

# RNN Recap

RNNs can have many forms as shown in the figure below:

![Screenshot from 2025-05-18 02-58-07](https://hackmd.io/_uploads/B1ah7s8Zel.png)

- **One to One:** Traditional neural network  
- **One to Many:** Image captioning  
- **Many to One:** Sentiment analysis  
- **Many to Many (fixed output size):** Named Entity Recognition (NER)  
- **Many to Many:** Machine translation  

We will focus on **many to many with variable length output**, often called **seq2seq**, and specifically on the **machine translation** problem.

## Sequence to Sequence Learning with Neural Networks (2014)

This was the first paper introducing the seq2seq model, developed by Google, which has gathered roughly **29,000 citations**.

![Screenshot from 2025-05-18 03-07-15](https://hackmd.io/_uploads/SJI6EsLbxx.png)

> In this paper, we present a general end-to-end approach to sequence
learning that makes minimal assumptions on the sequence structure. Our method
uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence
to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set

-- *Sequence to Sequence Learning with Neural Networks*

### High-Level Overview

At a very high level, an encoder-decoder model can be thought of as two blocks, the encoder and the decoder connected by a vector which we will refer to as the **context vector**.

![Screenshot from 2025-05-18 03-18-54](https://hackmd.io/_uploads/H1NKviI-xg.png)

**Encoder:** The encoder processes each token in the input-sequence. It tries to cram all the information about the input-sequence into a vector of fixed length i.e. the **context vector**.

**Context vector:** The vector is built in such a way that it's expected to encapsulate the whole meaning of the input-sequence and help the decoder make accurate predictions. 

**Decoder:** The decoder reads the context vector and tries to predict the target-sequence token by token.

In the paper, they used LSTMs in both the encoder and decoder.

> "Surprisingly, the LSTM did not suffer on very long sentences, despite the recent experience of other researchers with related architectures."

— *Sequence to Sequence Learning with Neural Networks*

Their model architecture looks like this:

![Screenshot from 2025-05-18 03-26-19](https://hackmd.io/_uploads/rkrSFjLZxg.png)

During training, they applied an unusual trick by **reversing the source sentences**.  
For example, instead of the input being:  
`"nice to meet you"`  
they used:  
`"you meet to nice"`

After this trick, the BLEU score improved, although they did not fully understand why.

> "While we do not have a complete explanation to this phenomenon, we believe that it is caused by the introduction of many short term dependencies to the dataset."

— *Sequence to Sequence Learning with Neural Networks*

**Fun fact:** They used a C++ implementation for the LSTM.

![Screenshot from 2025-05-18 03-34-24](https://hackmd.io/_uploads/Hk27jo8Wxg.png)

![Screenshot from 2025-05-18 03-34-55](https://hackmd.io/_uploads/rJbSsoUWxe.png)

![Screenshot from 2025-05-18 03-39-13](https://hackmd.io/_uploads/HkeAB3oL-xl.png)

[Encoder-Decoder Seq2Seq Models, Clearly Explained!!](https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b)

## RNN With Attention

RNNs suffer from vanishing gradients, which means the contribution of the first word to the prediction of the last token becomes very small.

To address this, the **attention mechanism** was introduced.

The first paper to introduce attention was in 2016, and it was also in the context of **Neural Machine Translation (NMT)**.


![Screenshot from 2025-05-18 03-49-29](https://hackmd.io/_uploads/HyWfynIbxx.png)

> A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus.

-- *NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE*

Attention mechanism helps to look at all hidden states from encoder sequence for making predictions unlike vanilla Encoder-Decoder approach.

![Screenshot from 2025-05-18 04-47-24](https://hackmd.io/_uploads/S1Grh3LZee.png)

The difference in preceding image explained: In a simple Encoder-Decoder architecture the decoder is supposed to start making predictions by looking only at the final output of the encoder step which has condensed information. On the other hand, attention based architecture attends every hidden state from each encoder node at every time step and then makes predictions after deciding which one is more informative.

But how does it decide which states are more or less useful for every prediction at every time step of decoder?

**Solution: Use a Neural Network to “learn” which hidden encoder states to “attend” to and by how much**

The context vector $c_i$ is computed as a weighted sum of the hidden states $h_j$:

$$c_i = \sum_j \alpha_{ij} h_j$$

where $\alpha_{ij}$ are the attention weights that we **learn** during training.

The attention score $e_{ij}$ is computed by a feedforward neural network $f$ applied to the concatenation of the decoder’s previous hidden state $s_{i-1}$ and the encoder hidden state $h_j$:


$$e_{ij} = f(s_{i-1} \| h_j)$$

The attention weights $\alpha_{ij}$ are then obtained by applying a softmax over these scores:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$


![Screenshot from 2025-05-18 04-49-44](https://hackmd.io/_uploads/HJfAnhUbgg.png)

This is a visualization for attention found in the paper 

![Screenshot from 2025-05-18 04-11-59](https://hackmd.io/_uploads/ryGrh38Zxe.png)


[A simple overview of RNN, LSTM and Attention Mechanism](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)

# Transformer Architecture

Transformers overcome RNNs' limitations in handling long-range dependencies and enable parallel processing of input sequences.

There are three main Transformer architectures:

1. **Encoder-only**: Used primarily for classification tasks (e.g., BERT).  
2. **Decoder-only**: Found in language models (e.g., GPT).  
3. **Encoder-decoder**: Used in Neural Machine Translation (NMT), which we will discuss today.


## Prepare Your Input 
Before you can use your data in a model, the data needs to be processed into an acceptable format for the model. A model does not understand raw text, images or audio. These inputs need to be converted into numbers
We prepare our text data using 3 steps 
1. Tokenize 
2. Get Input IDs

![Screenshot from 2025-05-17 04-31-13](https://hackmd.io/_uploads/Hklgx_rZee.png)

### 1) Tokenizer

A **tokenizer** starts by splitting text into tokens according to a set of rules.

There are several types of tokenizers:

1. **Word-based**  
2. **Character-based**  
3. **BPE (Byte Pair Encoding)** — used in GPT-2  
4. **WordPiece** — used in BERT  

State-of-the-art transformer models typically use **subword tokenizers** (BPE or WordPiece).

For a detailed comparison, check out [this article on tokenization](https://medium.com/data-science/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17).

---

#### a) Word-based Tokenization

This tokenizer splits text into words, usually by spaces.

Example:  
`["Let", "us", "learn", "tokenization."]`

**Pros**  
- Easy to construct and understand.

**Cons**  
- Large vocabulary size.  
- High out-of-vocabulary (OOV) rate if vocabulary is limited, leading to many unknown tokens.

---

#### b) Character-based Tokenization

This tokenizer splits text into individual characters.

Example:  
`["L", "e", "t", "u", "s", "l", "e", "a", "r", "n", "t", "o", "k", "e", "n", "i", "z", "a", "t", "i", "o", "n", "."]`

**Pros**  
- Very small vocabulary size.  
- No OOV tokens.

**Cons**  
- Longer sequences, which may increase computation.

> **Note:** Reducing vocabulary size by using character-level tokenization trades off with longer input sequences.

---

#### c) Subword-based Tokenization

This method splits text into meaningful subword units.

Example:  
`["Let", "us", "learn", "token", "ization."]`

Subword tokenization strikes a balance between word- and character-based methods by reducing vocabulary size and OOVs while keeping sequences reasonably short.

---
### Subword Tokenization Algorithms

There are several popular algorithms for subword-based tokenization:

1. **Byte Pair Encoding (BPE)**  
2. **WordPiece**

Both algorithms start by splitting the corpus into individual characters (or bytes) and then iteratively merge units to form larger subwords.

- **BPE:**  
  At each iteration, BPE merges the most frequent pair of adjacent symbols (characters or subwords) in the corpus. This process continues until a predefined vocabulary size or number of merges is reached.

- **WordPiece:**  
  Instead of simply merging the most frequent pairs, WordPiece merges subwords by maximizing the likelihood of the training data under a probabilistic model. 

---

### 2) Input IDs

After tokenization, the tokens are converted into numerical representations called **input IDs**.

For example, using GPT-2 model

- **Tokens:** `['Hello', 'ĠDeep', 'ĠLearning', '!']`  
- **Input IDs:** `[15496, 10766, 18252, 0]`

> **Note 1:** Input IDs are static, meaning the mapping from tokens to IDs does not change once the vocabulary is fixed.
> 
> **Note 2:** Each model has its own vocabulary and corresponding token-to-ID mapping.


---
## Attention Is All You Need (2017) 

**Now we can explore transformer architecture represented in attention is all you need paper and trace it step by step**
This paper was published in 2017 and has roughly 180,000 citations

![Screenshot from 2025-05-17 03-29-37](https://hackmd.io/_uploads/rJqHQYS-ll.png)


![Screenshot from 2025-05-17 05-44-10](https://hackmd.io/_uploads/HkRMddBbxx.png)

---
### Input Embedding 


> **"Nets are for fish;  
> Once you get the fish, you can forget the net.  
> Words are for meaning;  
> Once you get the meaning, you can forget the words."**  
> — *Zhuang Zhou*

👉 [For very detailed explanation see this chapter](https://web.stanford.edu/~jurafsky/slp3/6.pdf) 

![Screenshot from 2025-05-17 05-45-53](https://hackmd.io/_uploads/SkEOduBblg.png)


Our goal is to find a **representation of words** that captures their meaning and can be used in neural networks.  
We do this by assigning each word a **vector**, known as an **embedding**.

#### Idea 1: Use one hot encoding 

**What is one hot encoding?**
If our vocabulary consists of just 3 words:

```python
["Hello", "World", "NLP"]
````

We can represent each word as a one-hot vector:

* "Hello" → `[1, 0, 0]`
* "World" → `[0, 1, 0]`
* "NLP"   → `[0, 0, 1]`

In general, for a vocabulary of size $V$, each word is represented by a $V$-dimensional vector with a single 1 and the rest 0s.
(Sparse vector)

> Example: GPT-2 has a vocabulary of **50,257 tokens**, so each one-hot vector is **50,257-dimensional**!


A property we want from word embeddings is that **similar words are close together** in the embedding space.
Because it helps the model **generalize** better.

> For example, even if the model has never seen the word **"terrible"**, it may still understand it if it's close to a known word like **"horrible"** in the embedding space.

Since these two words have similar meanings, and their embeddings are close, the model can infer the meaning of unseen or rare words based on their proximity to known ones.


#### Idea 2: Co-occurrence Matrix

This matrix is thus of dimensionality $|V|×|V|$ and each cell records
the number of times the row (target) word and the column (context) word co-occur in some context in some training corpus.

![Screenshot from 2025-05-17 18-55-45](https://hackmd.io/_uploads/Hk39-NU-xe.png)

Now similar words are closer to each other but most enetries are 0 and the vector dimension is still = vocab size

Calculating raw frequencies is bad as there a lot of words like **"the"**, **"is"**, or **"a"** that appear in almost **every context**, regardless of meaning

To solve this issue we need to apply a weighting scheme 


#### Idea 3: TF-IDF
we define $w = tf * idf$ 

- **tf** (term frequency) = count of word *w* in the document  
- **document frequency** : number of documents a word *w* occur in 
- **idf** (inverse document frequency) = a measure of how unique the word is across documents


If a word appears in **few documents**, then **IDF is high** → this word is likely **important**.

If a word occurs **a lot in all documents**, then **IDF is low** → this word is likely **less important** even if its **TF is high**.

> **Note**: There is another algorithm for this weighting called pointwise mutual information (PMI)

#### Idea 4: Train Embeddings

Word2Vec is a technique for converting words into dense vectors. It aims to represent words in a continuous vector space, where semantically similar words are mapped to nearby points. These vectors capture information about the words based on the words around them.
Word2Vec uses two main approaches:

#### 1) Continuous Bag-of-Words (CBOW)

The continuous bag-of-words model predicts the central word using the surrounding context words, which comprises a few words before and after the current word.

**Example:**   
consider the sentence, 
```
“The cake was chocolate flavoured”
```

The model will then iterate over this sentence for different target words, such as:

```
“The ____ was chocolate flavoured” 
```
being inputs and “cake” being the target word.


#### 2) skipgram 

Skipgram works in the exact opposite way to CBOW. Here, we take an input word and expect the model to tell us what words it is expected to be surrounded by.

Taking the same example, with “cake” we would expect the model to give us 

```
“The”, “was”, “chocolate”, “flavoured” 
```

![Screenshot from 2025-05-17 19-35-57](https://hackmd.io/_uploads/S1xbiN8Zxl.png)


---

### Embeddings and Historical Semantics

Embeddings can also be a useful tool for studying how meaning changes over time,
by computing multiple embedding spaces, each from texts written in a particular
time period.

![Screenshot from 2025-05-17 18-50-11](https://hackmd.io/_uploads/S1m2e4IWeg.png)

---
### How Do We Measure Similarity?

There are two common ways to measure similarity between vectors:

1. **Dot Product**  
2. **Cosine Similarity**

---

### Dot Product


It is **maximized when the vectors point in the same direction** (i.e., are similar).
- Examples:

$[1, 2] \cdot [2, 4] = 1 \times 2 + 2 \times 4 = 10$

$[1, 2] \cdot [-2, -4] = 1 \times (-2) + 2 \times (-4) = -10$

However, dot product tends to be **larger for vectors with bigger magnitudes**, which can be misleading:

$[100, 200] \cdot [-1, 3] = 100 \times (-1) + 200 \times 3 = 500$

Even though the first pair \([1, 2]\) and \([2, 4]\) are more similar, their dot product (10) is less than 500.

To avoid this bias from magnitude, we **normalize** vectors and compute the cosine similarity:

$$\frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|}$$
which is the formula to get $cos(\theta)$

- If the vectors are **similar**, cosine similarity is close to **1**.
- If they are **orthogonal (unrelated)**, cosine similarity is **0**.
- If they are **opposite**, cosine similarity is **-1**.


---

### Embedding Arithmetic and Analogies

In embedding space, we often observe interesting relationships like:

$$
\text{king} - \text{man} \approx \text{queen} - \text{woman}
$$

This suggests that the vector difference between **king** and **man** is similar to the difference between **queen** and **woman**.

From this, we can approximate:

$$
\text{queen} \approx \text{king} - \text{man} + \text{woman}
$$


This analogy isn’t always perfectly accurate. For example, **"queen"** can have meanings beyond the feminine form of king, such as referring to a famous British band, which may introduce noise or ambiguity in the embedding space.

similarly:

$$
\text{Paris} - \text{France} + \text{Poland} \approx \text{Warsaw}
$$

Embeddings can reveal relationships between historical or cultural figures, such as Hitler and Mussolini

---

## Positional Encoding

![Screenshot from 2025-05-17 20-02-20](https://hackmd.io/_uploads/S1fV-SLbxx.png)


### Positional Encoding in Transformers

The **position of a word** in a sequence matters for understanding context.

* In **RNNs**, words are processed **sequentially**, so positional information is naturally preserved.
* In **Transformers**, input tokens are processed **in parallel**, so the model **does not inherently know the order** of words.

### Solution: Add Positional Encoding

To provide order information, we **add a positional encoding (PE) vector to the token embeddings**.

* Original embedding shape: $({seq}, d_{model})$

  * `seq`: sequence length
  * `dmodel`: embedding dimension (a hyperparameter, set to 512 in the paper)

* We compute the final input embedding as:

$$\tilde{E} = E + PE$$

Where:

* $E$ is the original embedding matrix $({seq}, d_{model})$
* $PE$ is the positional encoding matrix $({seq}, d_{model})$


### How to Calculate Positional Encoding (PE)

The positional encoding is a **function** designed to satisfy certain criteria:

* **Changes should be smooth and not too large**, so semantic meaning is preserved.
* **bounded** 
* **periodic**
* Each position has a **unique encoding**.


### The Sinusoidal PE Function

We use sine and cosine functions with different frequencies to encode position:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

This design balances encoding **position** while preserving **semantic information** in the embeddings.

### Learnable PE Function

Instead of fixed sinusoidal functions, the positional encoding vectors can also be **learnable parameters**

> We also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results. 
We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

—  *"Attention is All You Need"*

[Positional embeddings in transformers EXPLAINED](https://www.youtube.com/watch?v=1biZfFLPRSY&t=456s&ab_channel=AICoffeeBreakwithLetitia)

---

## Attention Mechanism

![Screenshot from 2025-05-17 21-43-04](https://hackmd.io/_uploads/ByD0uULbgl.png)

Before tackling multihead attention we can explain self attention first

### Self Attention 

```
The chicken didn't cross the road because it was tired
The chicken didn't cross the road because it was wide
```

Both sentences has the same embedding vector but after the last word the word `it` is not the same 
so our goal is to build contextual embeddings that depends on the sequence 

To calculate the relationship of each word with others in a sentence (the **attention score**), we start by defining three matrices:

* **Query (Q)**
* **Key (K)**
* **Value (V)**

These are projections of the embedding layer $E$:

$$E(seq,d_{model})$$

Projection matrices:

$$
W_Q, W_K, W_V ({d_{\text{model}} \times d_{\text{model}}})
$$

Then:

$$Q = E W_Q, \quad K = E W_K, \quad V = E W_V$$

Each of these matrices has shape:

$$Q, K, V ({{seq}, d_{\text{model}}})$$

### Intuition

* **Q (Query)**: What am I searching for?
* **K (Key)**: How do others describe themselves? (like titles or summaries)
* **V (Value)**: The actual content or information returned if the match is close enough.

### Computing Attention Scores

The attention score is a similarity measure between **Q** and **K**:

$$\text{score} = Q K^T$$

To prevent the dot product from growing too large in magnitude, we use **scaled dot product attention**:

$$\alpha = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{model}}}}\right)$$

Here, $\alpha$ represents the attention weights after softmax normalization.

$$\alpha (seq,seq)$$

![Screenshot from 2025-05-17 22-29-22](https://hackmd.io/_uploads/Hky6QvLZxx.png)


$$Attention = \alpha* V$$
$Attention (seq,d_{model})$ which is the same as $E$ we can picture it as we built stronger embedding layer 

$$Attention = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{model}}}}\right) * V$$


### Multi-Head Attention (MHA)

So far, we've discussed **one attention head**. You can think of one head as answering **one type of question**.

But we often want to ask **multiple questions** so we use **multiple attention heads**

We defined $d_k$ as the head size 

  $$
  d_k = \frac{d_{\text{model}}}{h}
  $$
* We **slice** Q, K, and V along the embedding dimension into $h$ smaller parts.

Each head performs attention **in parallel**, and their outputs are concatenated:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

And the output here is $(seq,d_{model})$ as previously 

### From the Paper

> "In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{\text{model}} / h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."

— *"Attention Is All You Need"*

## Add & Norm

![Screenshot from 2025-05-17 22-58-43](https://hackmd.io/_uploads/S1q20uUblx.png)

$T1 = MHA(x)$
$T2 = T1 + x$
$T3 = LayerNorm(T2)$
$T4 = FFN(T3)$
$T5 = T4 + T3$
H = $LayerNorm(T5)$

And these steps is repeates $N_x$ times 
$N_x = 6$ in the paper but you can change it

This is called a residual connection tries to solve vanishing gradients

Lower layers in the FFN typically capture shallow or surface-level patterns.
For example: noticing that many sentences end with the same word.

Higher layers move beyond syntax and surface patterns to extract semantic information—they begin to understand meaning and relationships between concepts.

### Layernorm

* Reduces training time
* Reduces bias
* Prevents weights explosion

However, **LayerNorm** can restrict learning capacity.
To fix this, **two learnable parameters** are introduced:

$$\hat{x} = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$


### LayerNorm vs BatchNorm

| Feature         | LayerNorm                       | BatchNorm                     |
| --------------- | ------------------------------- | ----------------------------- |
| Normalizes over | Features of a **single sample** | Features across the **batch** |
| Best for        | RNNs, Transformers              | CNNs and feedforward networks |

![Screenshot from 2025-05-18 00-38-13](https://hackmd.io/_uploads/ByWkfY8Zex.png)

[LayerNorm vs BatchNorm](https://medium.com/@florian_algo/batchnorm-and-layernorm-2637f46a998b)

---

### Causal Masking in Attention

![Screenshot from 2025-05-18 00-50-52](https://hackmd.io/_uploads/SJqkBF8bxl.png)

We **don’t want the model to attend to future words** during training.
To achieve this, we use a **causal mask**

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

* The mask is an upper-triangular matrix filled with $-\infty$ above the diagonal.
The softmax will zero out the future tokens.

$M = 
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0       & -\infty & -\infty \\
0 & 0       & 0       & -\infty \\
0 & 0       & 0       & 0
\end{bmatrix}$

## Cross Attention

In decoder we take Key and Value from encoder and Q from the decoder itself and this is called cross attention

## Last Linear Layer

Last linear layer job is to map our dimentions to be $(seq,\text{vocab size})$
and then we predict next word 

We have multiple decoding strategies we can cover them next session!!

[You can find them here](https://freedium.cfd/https://medium.com/@lmpo/mastering-llms-a-guide-to-decoding-algorithms-c90a48fd167b
)
