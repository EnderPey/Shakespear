# Shakespeare Language Model
- **These models are based on https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py and the rest of Andrej Karpathy's Zero to Hero series**

This repository contains a character-based and token language model trained on a dataset of Shakespeare's works. The model is built using the Transformer architecture, implemented in PyTorch.

## Overview

These models were trained on the "tinyshakespeare" dataset, which is a subset of Shakespeare's works. The dataset was split into 90% for training and 10% for validation.

The Transformer architecture is composed of a series of blocks that include multi-head self-attention and feedforward layers. The model also utilizes positional embeddings for each token, which allows the model to understand the order of the input characters.

Here are the key hyperparameters used for training the BigramModel:

 - batch_size = 64
 - block_size = 256
 - max_iters = 5000
 - learning_rate = 3e-4
 - n_embd = 384
 - n_head = 6
 - n_layer = 6
 - dropout = 0.2

The model was trained using the [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
 optimizer.

Here are the key hyperparameters used for training the GPTModel:

 - block_size: int = 64 
 - vocab_size: int = 50304 
 - n_layer: int = 4 
 - n_head: int = 4 
 - n_embd: int = 128 
 - dropout = 0.1
 - iterations = 4000

Cosine decay learning rate with max lr = 6e-4, min lr being 10 percent of that, and warmups steps being the first 25% of iterations.

The model was trained using the [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
 optimizer with weight decay of 0.1, betas=(0.9, 0.95),  and eps=1e-8.

## Components

### Token and Position Embeddings
The model starts by converting the input characters or tokens into embeddings using a token embedding table. Additionally, position embeddings are added to the token embeddings to help the model understand the order of the characters. The combined embeddings are then passed through the Transformer blocks.

### Transformer Blocks
Each Transformer block consists of two main components:

1. Multi-head self-attention: This allows the model to weigh the importance of different characters in the input sequence when predicting the next character. The self-attention mechanism computes attention scores (weights) by comparing the query, key, and value vectors. These scores are then used to aggregate the value vectors, resulting in an output that emphasizes the most relevant characters.
  
2. Feedforward layers: Each Transformer block also contains a feedforward sub-layer that consists of two linear layers with a ReLU activation function for the Bigram model and a GELU activation function in between them. This sub-layer helps the model learn more complex relationships between the input characters.
  
The output of each Transformer block is then passed to the next block, allowing the model to learn increasingly abstract relationships between the characters.

### Final Layer Normalization and Linear Projection
After passing through all the Transformer blocks, the embeddings undergo a final layer normalization to ensure their values are on a similar scale. The normalized embeddings are then passed through a linear layer that projects them onto the output vocabulary size, resulting in logits for each possible character.

## Training
The model is trained using a batched approach, where it processes multiple independent sequences in parallel. At each iteration, a batch of input sequences and their corresponding target sequences are sampled from the training data. The model computes the logits for the input sequences, and the loss is calculated using cross-entropy between the logits and the target sequences.

For the GPTmodel, I implemented batch accumelation where I simulated large match sizes by acumelating losses together. This allows me to use larger batch sizes and sequence lengths without using a larger memory and instead sacrifising it for time training.

The gradients of the loss with respect to the model's parameters are computed using backpropagation, and the optimizer updates the model's weights accordingly.

## Evaluation
During training, the model's performance is regularly evaluated on both the training and validation datasets. This helps to monitor the model's progress and detect any signs of overfitting. The evaluation process involves estimating the average loss over a fixed number of iterations for both the training and validation sets.

Here is the training and validation loss graph for the GPTmodel:
![alt text](https://github.com/EnderPey/Shakespeare/blob/a84ee6cac887661d7c82efef32f3970919b04144/GPT/loss_plot.png)


## Text Generation
The trained model can be used to generate text by providing a context as input and specifying the desired number of new tokens to generate. The model outputs logits for the next character, which are then converted to probabilities using the softmax function. A character is sampled from the probability distribution, and the process is repeated until the desired number of characters is generated.

The GPTmodel can also generate any number of sequence simultaneously.

I have example outputs for each model listed in the repositorie for anyone curious

## Challenges

 - Learning how to read documentation and api implementation so I could optimize the model to run on my Macbook M2 chip.

 -  Scaling down the GPTmodel in order to prevent overfitting and my validation losses from increasing.

 -  Chopping up the training in smaller batches by using checkpoints and loading the model back in to resume training

## Lessons Learned
  
### 1. Transformer architecture
Implementing the Transformer architecture helped gain a deeper understanding of the inner workings of self-attention, multi-head attention, and positional embeddings.

  
### 2. Character-level language modeling
 Working with a character-based dataset provided insights into the challenges and nuances of predicting the next character in a sequence, as opposed to word-level language modeling.

 ### 3. OpenAI API
 Working with the OpenAI api tokenizer in order to encode and decode my dataset allowing the model to process and understand a lot more.
 
### 3. Dataset splitting and evaluation
 Splitting the dataset into training and validation sets, and regularly evaluating the model's performance on both sets, helped monitor the model's generalization capabilities and prevent overfitting.

  
### 4. Text generation
Implementing the model's text generation functionality provided experience in working with the softmax function, sampling from a probability distribution, and concatenating the generated tokens to create the final output.
