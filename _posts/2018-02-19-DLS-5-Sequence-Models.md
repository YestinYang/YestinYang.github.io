---
layout: post
title: Sequence Models - Deep Learning Specialization 5
key: 20180219
tags:
  - Coursera
  - Notes
  - Study
  - Deep Learning
  - Data Science
lang: en
mathjax: true
mathjax_autoNumber: true
---

deeplearning.ai by Andrew Ng on Coursera

## W1: Recurrent Neural Networks

### Building Sequence Model

**Notation:**



![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517471514373.png){:.border}

**Model Architecture:**
- Why standard network works not well?
	- Inputs, outputs can be different lengths in different samples
	- Doesn't share features learned across different positions of text
		- CNN learns from one part of the image and generalize to other parts, where each filter represents one kind of learning object and convolution apply it across the image
		- RNN is also like a 'filter' swapping through the sequence data
	-  Size of one-hot encoded input is too large to handle
- Uni-directional RNN (get the information from past steps only)
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517470831372.png){:.border}
	- ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517472116692.png){:.border}

### Types of RNN

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517449657550.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517472966792.png){:.border}

### Language Model and Sequence Generation

**Purpose:** exam the probability of sentences



![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517531029321.png){:.border}

**Training the model:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517531573532.png){:.border}

**Sampling Novel Sequence:** to get a sense of model prediction, after training
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517539671522.png){:.border}

**Character-level Language Model:** can handle unknown words but much slower

### Address Vanishing Gradient by GRU / LSTM

> Also has exploding gradient problem, but it is easier to be solved by gradient clipping
> ![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517732026577.png){:.border}

**Vanishing Gradient:**
- Like very deep neural network, for a very deep RNN, the gradient for earlier layer is too small to affect those parameters
- In practice, it means that the result of later layers are hard to be strongly influenced by earlier layers. In other words, RNN tend not to be good at capturing long-ranged dependencies. 
	- can be understood as with only "short-term" memory
- Sentence example: use was or were? 
	- The `cat`, which ... [long parenthesis], `was` full.
	- The `cats`, which ... [long parenthesis], `were` full.

**Gated Recurrent Unit (GRU):** simplified from LSTM
- Basic idea:
	- conventional RNN uses linear weighted past information, so by going through large number of layers, the information from earlier layer is 'weighted' too many times and left nearly none.
	- GRU use a gate to control 'update or not update' each element in activation function, so that if the old information is not 'significant' enough ($\Gamma_u$), it will be replaced by new information
- Compared to LSTM, it represents Update Gate and Forget Gate in LSTM, by $\Gamma_u$ and $1-\Gamma_u$
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517560446857.png){:.border}

**Long Short Term Memory (LSTM):** more general than GRU
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517577362417.png){:.border}

### Bidirectional RNN

**Condition of Application:** need entire sentence to get the result
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517621966940.png){:.border}

### Deep RNN
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517622524871.png){:.border}

## W2: Natural Language Precessing & Word Embeddings

### Introduction to Word Embeddings

> One-hot representation of words treats each word as a thing unto itself, which is hard for algorithm to generalize the cross words

**Word Embedding:** Featurized Representation to find out words with similar properties
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1517975903334.png){:.border}

**Transfer Learning with Word Embedding:**
1. Learn word embeddings from large text corpus (1-100B words), or download pre-trained embedding.
2. Transfer embedding to new task with smaller training set (maybe ~100k words).
3. Optional if step 2 has enough large dataset: continue to finetune the word embeddings with new data.

**Difference between Encoding and Embedding:**
- Encoding in face recognition is a algorithm can use any image as input, and then find out the characteristic on them
- Embedding in NLP cannot handle any unknown vocabulary from input

**Analogy Reasoning with Word Vector:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518010060113.png){:.border}

**Embedding Matrix Notation:**



![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518010564486.png){:.border}

### Learning Word Embeddings: Word2vec & GloVe

**Word2Vec (Context and Target Pair):**
> This is to learn word embedding matrix, not to predict

- Main problem is that softmax is computational expensive
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518224562037.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518226993143.png){:.border}

**Negative Sampling:** similar but more efficient than skip-grams by transforming softmax to binary classification
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518231565523.png){:.border}

**GloVe Word Vectors:** 



![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518257540649.png){:.border}

### Application using Word Embeddings

**Sentiment Classification:**
> With word embeddings, only moderate size of labeled training dataset can build good model

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518260803792.png){:.border}

**Debiasing Word Embeddings:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518345233626.png){:.border}

## W3: Sequence Models & Attention Mechanism

### Seq2Seq with Encoder + Decoder Architecture

**Difference between Language Model and Seq2seq:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518658834599.png){:.border}

**Picking the Most Likely Sentence:**
- Why not greedy search, which picks each ONE word with highest probability at a time?
	- The result is affected by the popularity of each word, like 'is going to visit' is more common compared to 'is visiting' but worse translation
	- The final sequence is not the sequence with highest probability
- Why not considering the probability of the whole sequence?
	- computational expensive, a sequence with 10 words selecting from 10k vocabulary list has $10,000^{10}$ combinations
- `Beam Search`: approximation algorithm; not guarantee highest prob output
	1. take beam width = 3 as example
	2. selecting 3 results in 1st stage with top 3 highest prob
	3. feed these 3 results as input of 2nd stage and find out the 3 results with top 3 highest prob among $10,000^{3}$ combinations
	4. continue Step 3 until end of the sequence
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518661280496.png){:.border}

**Improving Beam Search:** length normalization
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518833018826.png){:.border}

**Error Analysis on Beam Search:**
- Compare the probability of the translation from human and algorithm, to identify the error comes from RNN or beam search
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518835854760.png){:.border}

**Bleu Score:** bilingual evaluation understudy
> Evaluate 'accuracy' of a model predicting multiply equally good answers, being a substitute for human evaluating each output
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518855087684.png){:.border}

### Attention Model

> Counter the problem of long sentence, which requires the ability of memory but not badly need a NN to do this kind of job.
> Instead of 'remembering' the whole sentence and then generate output sequence, only focus on nearby words corresponding to each time step, while the range of 'nearby' is learnt by gradient descent, and 'adjusted' corresponding to every different case (input sequence).
>> Ability of attention adaptation to input is because $\alpha$ is determined by both output LSTM at t-1 and input LSTM at t

![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518863490598.png){:.border}
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1519013067406.png){:.border}

### Speed Recognition - Audio Data

**Approach 1: Attention Model:**
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518864439681.png){:.border}

**Approach 2: CTC Model:** Connectionist Temporal Classification
![Alt text](https://github.com/YestinYang/YestinYang.github.io/raw/master/screenshots/2018-02-19_1518864757190.png){:.border}
