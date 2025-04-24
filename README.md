# Context-Free Grammar (CFG) Parser

This project implements a natural language parser using Context-Free Grammar (CFG) rules and a trained neural network model for Part-of-Speech (POS) tagging. The parser can analyze sentences and generate parse trees using both top-down and bottom-up parsing strategies.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Implementation Details](#implementation-details)
- [Dependencies](#dependencies)
- [Setup and Usage](#setup-and-usage)
- [Visualization](#visualization)
- [Training Process](#training-process)
- [Text Preprocessing](#text-preprocessing)
- [Grammar Rules](#grammar-rules)
- [Future Improvements](#future-improvements)
- [Algorithms](#algorithms)

## Overview

The CFG Parser is a web application that:

1. Takes a sentence as input
2. Uses a trained neural network to tag each word with its part of speech
3. Applies CFG rules to parse the sentence
4. Generates a visual parse tree using Graphviz
5. Supports both top-down and bottom-up parsing strategies

## Components

### 1. Neural Network Model (`parser_trainer.py`)

- Implements a simple neural network for POS tagging
- Uses word embeddings to capture semantic relationships
- Trained on the Penn Treebank dataset
- Outputs probability distribution over POS tags for each word

### 2. Parser Implementation (`app.py`)

- Flask web application serving the parser
- Implements both top-down and bottom-up parsing strategies
- Handles sentence preprocessing and tokenization
- Manages visualization generation

### 3. Grammar Rules

The parser uses a context-free grammar with the following rules:

```python
GRAMMAR = {
    'S': [['NP', 'VP'], ['DT', 'S']],
    'NP': [['DT', 'NN'], ['PRP'], ['NNP'], ['DT', 'S']],
    'VP': [['VBD', 'NP'], ['VBD', 'PP'], ['VBZ', 'NP'], ['VBZ', 'PP']],
    'PP': [['IN', 'NP']],
}
```

## Implementation Details

### Parsing Strategies

#### Top-Down Parsing

- Starts with the start symbol 'S'
- Expands non-terminals according to grammar rules
- Backtracks when a path fails
- More intuitive but can be less efficient

#### Bottom-Up Parsing

- Starts with the input words
- Reduces sequences of symbols according to grammar rules
- Builds the parse tree from leaves to root
- More efficient but can be less intuitive

### Neural Network Architecture

The parser uses a custom neural network architecture (`SimpleParser`) implemented in PyTorch with the following components:

1. **Embedding Layer**

   - Converts words to dense vectors
   - Input size: vocabulary size
   - Output size: 100 dimensions
   - Handles out-of-vocabulary words with an `<unk>` token

2. **LSTM Layer**

   - Bidirectional Long Short-Term Memory network
   - Hidden dimension: 128
   - Captures sequential dependencies in both directions
   - Uses dropout (0.3) for regularization
   - Processes input sequences in both forward and backward directions

3. **Linear Layer**
   - Maps LSTM output to POS tags
   - Input size: hidden_dim \* 2 (due to bidirectional LSTM)
   - Output size: number of POS tags
   - Produces probability distribution over possible tags

#### Training Details

- **Dataset**: Penn Treebank from NLTK
- **Loss Function**: Cross-entropy loss with padding token ignored
- **Optimizer**: Adam optimizer
- **Batch Size**: 32
- **Number of Epochs**: 10
- **Training Metrics**: Loss and accuracy tracked and visualized
- **Regularization**: Dropout (0.3) to prevent overfitting

The model takes a sequence of words as input and outputs a probability distribution over POS tags for each word in the sequence. This is then used by the parser to build the parse tree according to the grammar rules.

## Dependencies

### Core Libraries

- **Flask**: Web framework for serving the parser
- **NLTK**: Natural Language Toolkit for text processing
- **PyTorch**: Deep learning framework for the neural network
- **NetworkX**: Graph manipulation library
- **Matplotlib**: Plotting library for visualization

### Visualization

- **Graphviz**: Graph visualization software
  - Used to generate parse tree visualizations
  - Provides clear hierarchical representation of sentence structure
  - Supports automatic layout and styling

## Setup and Usage

1. Install dependencies:

```bash
pip install flask nltk torch networkx matplotlib
```

2. Install Graphviz:

- Windows: Download and install from [Graphviz website](https://graphviz.org/download/)
- Linux: `sudo apt-get install graphviz`
- macOS: `brew install graphviz`

3 . Download NLTK data:

```python
import nltk
nltk.download('treebank')
nltk.download('punkt')
```

4. Run the application:

```bash
python app.py
```

5. Access the web interface at `http://localhost:5000`

## Training Process

### Dataset

The parser uses the Penn Treebank dataset from NLTK, which is a widely-used annotated corpus of English text. The dataset provides:

1. **Data Structure**

   - Raw sentences
   - Parse trees with POS tags
   - Syntactic structure annotations

2. **POS Tags Used**

   - Basic tags: NN (noun), VB (verb), DT (determiner)
   - Verb forms: VBD, VBZ, VBP, VBG
   - Noun types: NN, NNS, NNP, NNPS
   - Other categories: PRP (pronoun), IN (preposition), etc.
   - Special tokens: `<pad>` for padding, `<unk>` for unknown words

3. **Data Processing**

   - Words are converted to lowercase
   - Special tokens are added for unknown words and padding
   - Data is shuffled before training
   - Sequences are padded to the same length for batch processing
   - Vocabulary and tag mappings are created and saved

4. **Dataset Size**
   - Contains approximately 40,000 sentences
   - Includes diverse sentence structures
   - Covers various domains and writing styles

### Training Steps

1. Load and preprocess the dataset
2. Create word and tag vocabularies
3. Train the neural network
4. Save the trained model and vocabularies

## Text Preprocessing

The parser implements several text preprocessing steps to ensure consistent and effective processing of natural language input:

### 1. Word Level Preprocessing

- **Lowercasing**: All words are converted to lowercase to ensure consistent processing
- **Special Tokens**:
  - `<unk>` token for handling out-of-vocabulary words
  - `<pad>` token for batch processing and sequence padding
- **Vocabulary Creation**:
  - Word-to-index mappings are created from the training corpus
  - Vocabulary size is determined by the unique words in the dataset
  - Mappings are saved for use in the web interface

### 2. POS Tag Preprocessing

- **Tag Set Creation**: Comprehensive tag set derived from Penn Treebank
- **Tag Mapping**: Model's POS tags are mapped to grammar-compatible tags:
  - 'NNS' → 'NN' (plural nouns)
  - 'VBP' → 'VBZ' (present tense verbs)
  - 'NNPS' → 'NNP' (plural proper nouns)
- **Special Tags**: Padding tag added for batch processing

### 3. Sentence Level Preprocessing

- **Tokenization**: Input sentences are split into individual words
- **Sequence Padding**:
  - Sequences are padded to ensure uniform length
  - Enables efficient batch processing
- **Data Shuffling**: Training data is shuffled to prevent order bias

### 4. Data Structure Preprocessing

- **Vocabulary Creation**: Word and tag mappings are created and saved
- **Index Conversion**: Words are converted to indices for neural network input
- **Mapping Storage**: Mappings are saved as JSON files for web interface use

These preprocessing steps are implemented in:

- `parser_trainer.py`: Initial dataset preprocessing and vocabulary creation
- `app.py`: Real-time preprocessing of user input sentences

The preprocessing pipeline ensures:

- Consistent input format for the neural network
- Graceful handling of unknown words
- Efficient batch processing
- Proper mapping between model output and grammar rules

## Grammar Rules

The grammar rules define how different parts of speech can combine to form valid sentences:

- **S** (Sentence): Can be an NP followed by a VP, or a DT followed by an S
- **NP** (Noun Phrase): Can be a DT followed by an NN, a PRP, an NNP, or a DT followed by an S
- **VP** (Verb Phrase): Can be a VBD or VBZ followed by an NP or PP
- **PP** (Prepositional Phrase): Can be an IN followed by an NP

## Future Improvements

1. **Enhanced Grammar**

   - Add support for more complex sentence structures
   - Include rules for questions and commands
   - Support for compound sentences

2. **Improved Model**

   - Larger training dataset
   - More sophisticated neural network architecture
   - Better handling of unknown words

3. **Visualization**

   - Interactive parse tree visualization
   - Support for different tree layouts
   - Custom styling options

4. **Performance**
   - Optimize parsing algorithms
   - Add caching for frequently parsed sentences
   - Support for batch processing

## License

This project is open source and available under the MIT License.

## Algorithms

### 1. Neural Network Algorithms

#### Word Embeddings

- Converts words to dense vector representations
- Captures semantic relationships between words
- Handles out-of-vocabulary words through an `<unk>` token
- Uses a vocabulary-based embedding layer

**Formula**:

```
E(w) = W_e * one_hot(w)
```

where:

- `E(w)`: Embedding vector for word w
- `W_e`: Embedding matrix of size (vocab_size × embedding_dim)
- `one_hot(w)`: One-hot encoding of word w

#### Bidirectional LSTM

- Processes input sequences in both forward and backward directions
- Captures long-range dependencies in text
- Uses dropout (0.3) for regularization
- Combines forward and backward hidden states for better context understanding

**Formulas**:

```
Forward LSTM:
i_t = σ(W_xi * x_t + W_hi * h_{t-1} + b_i)
f_t = σ(W_xf * x_t + W_hf * h_{t-1} + b_f)
o_t = σ(W_xo * x_t + W_ho * h_{t-1} + b_o)
g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)

Backward LSTM:
Same as above but processing sequence in reverse

Final Output:
h_t = [h_t_forward; h_t_backward]
```

where:

- `σ`: Sigmoid activation function
- `⊙`: Element-wise multiplication
- `W_*`: Weight matrices
- `b_*`: Bias vectors
- `h_t`: Hidden state at time t
- `c_t`: Cell state at time t

#### Training Algorithm

- Adam optimizer for parameter updates
- Cross-entropy loss function
- Batch processing with padding
- Gradient descent with backpropagation

**Formulas**:

```
Cross-Entropy Loss:
L = -Σ(y * log(p) + (1-y) * log(1-p))

Adam Optimizer:
m_t = β₁ * m_{t-1} + (1-β₁) * g_t
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

where:

- `y`: True label
- `p`: Predicted probability
- `g_t`: Gradient at time t
- `β₁, β₂`: Decay rates (typically 0.9 and 0.999)
- `α`: Learning rate
- `ε`: Small constant for numerical stability

### 2. Parsing Algorithms

#### Top-Down Parsing (Recursive Descent)

- Starts with the start symbol 'S'
- Recursively expands non-terminals according to grammar rules
- Uses backtracking when a path fails
- Builds parse tree from root to leaves
- More intuitive but can be less efficient for complex sentences

#### Bottom-Up Parsing (Shift-Reduce)

- Starts with input words
- Reduces sequences of symbols according to grammar rules
- Builds parse tree from leaves to root
- More efficient for many sentence structures
- Uses a shift-reduce approach to combine symbols

### 3. Visualization Algorithm

- Uses Graphviz for tree visualization
- Implements a recursive tree traversal algorithm
- Generates DOT language representation of parse trees
- Converts DOT to PNG for display
