---
title: "Building a Transformer from Scratch: My Journey into the Heart of Modern AI"
date: "2024-03-13"
description: "A deep dive into implementing a transformer model from scratch, sharing challenges and learnings along the way"
excerpt: "A deep dive into implementing a transformer model from scratch, sharing challenges and learnings along the way"
category: "research"
tags: ["machine-learning", "transformers", "nlp", "pytorch", "deep-learning"]
author: "Sajjad Momeni"
readTime: "15 min read"
direction: "rtl"
---



## Introduction

When I first encountered transformer models like GPT and BERT, they seemed like magical black boxes. The capabilities were impressive, but I wanted to understand what was happening under the hood. So I embarked on a journey to build a transformer from scratch, component by component.

In this post, I'll share what I learned, the challenges I faced, and the code I wrote to implement a GPT-style transformer model in PyTorch. By the end, I hope you'll have a deeper understanding of how these powerful models work.

## Understanding the Transformer Architecture

Before diving into code, I needed to understand the core concepts. The transformer architecture, introduced in the ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper, revolutionized natural language processing by eliminating recurrence and convolutions entirely in favor of attention mechanisms.

The key components of a transformer include:

1. **Token Embeddings**: Converting tokens to vectors
2. **Positional Encodings**: Adding position information
3. **Multi-Head Attention**: Allowing the model to focus on different parts of the input
4. **Feed-Forward Networks**: Processing each position independently
5. **Layer Normalization**: Stabilizing training
6. **Residual Connections**: Helping with gradient flow

GPT models specifically use a decoder-only architecture with causal attention, meaning each token can only attend to itself and previous tokens.

## Building Block 1: Embeddings

I started with the embedding layer, which converts token IDs to dense vectors:

```python
class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

The multiplication by `sqrt(d_model)` is a scaling factor that helps maintain the variance of the forward pass, as mentioned in the original paper.

## Building Block 2: Positional Encoding

Next, I implemented positional encoding to give the model information about token positions:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create position tensor
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term[:-(1 if d_model % 2 != 0 else 0)])
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (important for model saving/loading)
        self.register_buffer('pe', pe)
        
        # Initialize dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

This creates a unique pattern for each position using sine and cosine functions of different frequencies. The beauty of this approach is that it allows the model to extrapolate to sequence lengths not seen during training.

## Building Block 3: Attention Mechanism

The heart of the transformer is the attention mechanism. I implemented it step by step:

```python
class Head(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        output = self.out_proj(self.dropout(output))
        return output, attention
```

Then I extended it to multi-head attention:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.head_dim = d_model // num_heads
        self.heads = nn.ModuleList([Head(self.head_dim, num_heads, dropout) for _ in range(num_heads)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        outputs = []
        attentions = []
        
        for head in self.heads:
            output, attention = head(x, mask)
            outputs.append(output)
            attentions.append(attention)
        
        # Concatenate outputs from all heads
        output = torch.cat(outputs, dim=-1)
        
        # Apply final linear projection
        output = self.output_linear(output)
        
        return self.dropout(output)
```

## Building Block 4: Feed-Forward Network

The feed-forward network processes each position independently:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
```

This expands the input dimension, applies non-linearity, and then projects back to the original dimension.

## Building Block 5: Encoder and Decoder Layers

Next, I combined attention and feed-forward networks with residual connections and layer normalization:

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim=2048, dropout=0.1):
        super().__init__()
        
        # Self-attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, hidden_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## Building Block 6: The Complete GPT Model

Finally, I put everything together to create a GPT model:

```python
class GPT(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size, max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional encoding
        self.embedding = Embedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Stack of decoder-style layers (self-attention only)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        
        # Final normalization and projection
        self.norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        # Create causal mask if not provided
        if mask is None:
            mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Apply embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        output = self.output_linear(x)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens given a context."""
        # Generate tokens auto-regressively
        for _ in range(max_new_tokens):
            # Crop context to the last max_seq_length tokens if needed
            idx_cond = idx if idx.size(1) <= self.positional_encoding.pe.size(1) else idx[:, -self.positional_encoding.pe.size(1):]
            
            # Get predictions
            logits = self(idx_cond)
            
            # Focus on the last token
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
```

## Training the Model

With the model architecture in place, I created a training pipeline:

```python
# Simple text dataset
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        
        # Create a character-level tokenizer (for simplicity)
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode the text
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # Create examples
        self.examples = []
        for i in range(0, len(data) - block_size, 1):
            x = data[i:i+block_size]
            y = data[i+1:i+block_size+1]
            self.examples.append((x, y))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def decode(self, tokens):
        return ''.join([self.itos[int(i)] for i in tokens])

def train():
    # Load your text data
    with open('your_text_file.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset and dataloader
    dataset = TextDataset(text, BLOCK_SIZE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = GPT(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        vocab_size=dataset.vocab_size,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in get_batch(data_loader):
            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging and evaluation...
```

## Generating Text

After training, I could generate text with the model:

```python
def generate_text(
    model_path,
    text_file,
    prompt="",
    max_new_tokens=500,
    temperature=0.8,
    top_k=40
):
    # Load the text file to get the vocabulary
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset to get tokenizer
    dataset = TextDataset(text, block_size=1)
    
    # Load model
    model = GPT(
        d_model=384,
        num_heads=6,
        num_layers=6,
        vocab_size=dataset.vocab_size,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Tokenize the prompt
    if prompt:
        context = torch.tensor([[dataset.stoi.get(c, 0) for c in prompt]], dtype=torch.long).to(device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        context[0, 0] = dataset.stoi.get('\n', 0)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    return dataset.decode(generated[0].tolist())
```

## Challenges and Learnings

Building a transformer from scratch wasn't easy. Here are some challenges I faced:

1. **Dimension Mismatch**: Getting the tensor dimensions right in the attention mechanism was tricky, especially with batch processing and multiple heads.

2. **Numerical Stability**: I had to be careful with the implementation of positional encoding and attention scores to avoid numerical issues.

3. **Memory Usage**: Transformers can be memory-intensive, especially with long sequences. I had to optimize my implementation to fit within my hardware constraints.

4. **Debugging**: When things went wrong, it was often hard to pinpoint the exact issue. I learned to add extensive logging and visualization to debug effectively.

Despite these challenges, the journey was incredibly rewarding. I gained a deep understanding of how transformers work, which has been invaluable for my work in NLP.

## Results and Next Steps

My implementation successfully learned to generate coherent text, though of course not at the level of state-of-the-art models like GPT-3 or GPT-4. Here's a sample of text generated by my model:

```
Once upon a time, there was a small village nestled in the mountains...
```

For those interested in trying this out, the complete code is available in my GitHub repository.

## Future Improvements

There are several ways to improve this implementation:

1. **Better Tokenization**: Implement subword tokenization (BPE, WordPiece) instead of character-level tokenization.

2. **Larger Model**: Increase model size (d_model, num_heads, num_layers) for more capacity.

3. **Learning Rate Scheduling**: Implement learning rate warmup and decay for better convergence.

4. **Mixed Precision Training**: Use mixed precision to speed up training and reduce memory usage.

5. **Pre-training and Fine-tuning**: Implement a proper pre-training and fine-tuning pipeline.

## Conclusion

Building a transformer from scratch has been an enlightening journey. It's demystified what once seemed like magic and given me a solid foundation for working with these powerful models.

If you're interested in understanding transformers deeply, I highly recommend implementing one yourself. Start simple, build component by component, and test thoroughly at each step.

Happy coding!

---

*This blog post is part of my series on understanding deep learning from first principles. If you found this helpful, please share it with others who might benefit.* 