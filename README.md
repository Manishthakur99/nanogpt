# Character-Level Language Models

This repository is a compact, educational progression from a simple character-level language model to a small decoder-only Transformer. It uses PyTorch and a plain-text corpus in `input.txt`.

The two scripts are standalone training and generation programs:

| Script | Model | Context window | Batch size | Training steps |
| --- | --- | ---: | ---: | ---: |
| `bigram.py` | Embedding + linear prediction baseline | 8 characters | 32 | 3,000 |
| `v2.py` | Causal multi-head self-attention Transformer | 32 characters | 16 | 5,000 |

Both scripts read `input.txt` from the current working directory, train from scratch, print periodic training and validation losses, then sample and print generated text. They do not save model checkpoints.

## Architecture

### Shared character-level data pipeline

The corpus is read as a single string. The scripts construct a vocabulary from the sorted set of characters that occur in that string, then create two mappings:

- `stoi`: character to integer token ID
- `itos`: integer token ID back to character

Each character is one token; there is no subword tokenizer. The complete text is encoded as integer IDs and split in order: the first 90% is training data and the final 10% is validation data.

For each batch, random starting positions are sampled from the selected split. An input sequence contains `block_size` consecutive characters, and its target is the same sequence shifted one character forward. At every position, the model therefore learns to predict the next character.

```text
Text:   "to be or not"
Input:  "to be or n"
Target: "o be or no"
```

The implementation batches these examples into tensors of shape `(batch_size, block_size)` and moves them to CUDA when available, otherwise to CPU.

### `bigram.py`: embedding baseline

The baseline maps each input character ID to a learned token embedding, adds a learned embedding for its position within the fixed window, and applies a linear layer to produce vocabulary logits:

```text
character IDs -> token embeddings --+
                                     +-> sum -> linear vocabulary head -> logits
positions -----> position embeddings+
```

Cross-entropy loss is computed over all positions when targets are provided. Generation repeatedly samples a character from the final position's softmax distribution and appends it to the sequence.

This is deliberately a small baseline, not a Transformer. It has no attention layer: a position's prediction is made from its token embedding and position embedding rather than from an attention-based aggregation of earlier tokens. Its generation loop also does not crop the context to `block_size`; the learned position table limits the supported input length.

### `v2.py`: causal Transformer

`v2.py` uses the same character vocabulary, batching, loss, and autoregressive sampling approach, with a deeper network between the embeddings and vocabulary head:

```text
character IDs + learned positions
              |
              v
      4 × Transformer block
       +-- pre-normalized causal multi-head self-attention
       |    +-- residual connection
       +-- pre-normalized feed-forward network
            +-- residual connection
              |
              v
        final LayerNorm
              |
              v
      linear vocabulary head
              |
              v
     next-character logits
```

The configured model width is 64, with 4 attention heads and 4 Transformer blocks. Each attention head independently computes learned queries, keys, and values. Query-key dot products produce attention scores; the implementation scales these scores by the inverse square root of the model width. A lower-triangular causal mask prevents a position from attending to future characters. A softmax converts the permitted scores into weights, which are used to combine value vectors. The head outputs are concatenated, projected back to the model width, and passed through dropout.

Each block has two pre-normalized residual sublayers:

1. Causal multi-head self-attention, for information exchange between positions.
2. A position-wise feed-forward network: linear expansion from 64 to 256 features, ReLU, projection back to 64 features, and dropout.

After the blocks, a final layer normalization and linear projection produce one vocabulary-sized logit vector per position. During generation, `v2.py` crops the input context to the most recent 32 characters before each prediction.

## Training configuration

The hyperparameters are defined at the top of each script and can be changed there.

| Setting | `bigram.py` | `v2.py` |
| --- | ---: | ---: |
| `batch_size` | 32 | 16 |
| `block_size` | 8 | 32 |
| `max_iters` | 3,000 | 5,000 |
| `eval_interval` | 300 | 100 |
| `eval_iters` | 200 | 200 |
| `learning_rate` | 0.01 | 0.001 |
| `n_embd` | 32 | 64 |
| `n_head` / `n_layer` | — | 4 / 4 |
| `dropout` | — | 0.0 |

Both scripts seed PyTorch with `1337`, optimize cross-entropy using AdamW, and estimate loss on randomly sampled batches from both splits. `v2.py` evaluates at step 0, then at each interval and the final step; `bigram.py` evaluates at step 0 and each interval. The reported validation loss is for monitoring and is not used for optimization.

## Requirements

- Python 3
- PyTorch
- `input.txt` in the directory from which the script is run

Install PyTorch using the instructions for your operating system and compute platform at [pytorch.org](https://pytorch.org/get-started/locally/).

## Run

From the repository root:

```bash
python bigram.py
```

Or run the Transformer:

```bash
python v2.py
```

Each script trains a new model before generating text. The sampling prompt starts from a one-token sequence whose ID is zero (the first character in the sorted vocabulary); no dedicated beginning-of-sequence token is defined.

## Scope and limitations

- This is an educational, character-level implementation, not a production language-model training framework.
- Training and generation happen in one script execution. There are no command-line options, checkpointing, resume support, or separate inference mode.
- The vocabulary is built from the entire corpus before the train/validation split, so validation characters are already represented in the vocabulary.
- The train/validation split is a contiguous split of the text, not a shuffled or document-level split.
- Generation is stochastic multinomial sampling with no temperature, top-k, or top-p controls.
- `bigram.py` expects context lengths compatible with its positional embedding table; `v2.py` explicitly limits each model input to its configured context window.
