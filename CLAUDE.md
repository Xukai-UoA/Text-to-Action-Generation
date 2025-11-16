# Text-to-Motion-Generation Repository Guide

## Project Overview

This is a **Text-to-Motion Generation** system that converts natural language descriptions into realistic human motion sequences. The project implements two complementary architectures:

1. **Seq2Seq Model** - A sequence-to-sequence encoder-decoder architecture for baseline motion generation
2. **GAN Model** - A Generative Adversarial Network that refines motion quality and introduces diversity through adversarial training

The system uses PyTorch and is designed to generate 32-frame motion sequences (24-dimensional pose vectors) from natural language text input.

---

## Directory Structure

```
Text-to-Motion-Generation/
├── README.md                    # Basic project description
├── CLAUDE.md                    # This file - AI assistant guide
│
├── model/                       # Core model implementations
│   ├── __init__.py
│   ├── seq2seq_structure.py     # Seq2Seq model architecture
│   ├── seq2seq_trainer.py       # Seq2Seq training loop
│   ├── seq2seq_tester.py        # Seq2Seq inference/testing
│   ├── gan_structure.py         # GAN model architecture
│   ├── gan_trainer.py           # GAN training loop
│   └── seq2seq_enhanced_visualization.py  # Advanced motion visualization
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── my_functions.py          # Word2Vec loading, metadata handling
│   └── process_data.py          # Data preprocessing pipeline
│
├── seq2seq_pretrain.py          # Entry point: Seq2Seq training
├── gan_train.py                 # Entry point: GAN training
├── test_seq2seq_simple.py       # Entry point: Simple inference testing
├── test_seq2seq_enhanced.py     # Entry point: Enhanced visualization testing
│
├── data/                        # [NOT INCLUDED - User must provide]
│   ├── metadata.npz             # Preprocessed training data
│   ├── mean_pose.mat            # Initial pose for generation
│   ├── GoogleNews-vectors-negative300.bin  # Word2Vec embeddings
│   ├── pose/                    # Raw pose .mat files
│   └── script/                  # Text descriptions
│
├── seq2seq_model/               # [NOT INCLUDED - Generated during training]
│   └── model_epoch_500.pth      # Pretrained Seq2Seq checkpoint
│
├── gan_model/                   # [NOT INCLUDED - Generated during training]
│   └── model_epoch_*.pth        # GAN checkpoints
│
└── test_results/                # [OPTIONAL - Generated during testing]
    ├── *.png                    # Visualization images
    ├── *.gif                    # Animation GIFs
    └── *.npy                    # Generated motion data

```

---

## Key Data Formats & Dimensions

### Training Data (metadata.npz)
```
train_action:      [num_data, 24, 32]    # Pose vectors (24D) over 32 frames
train_script:      [num_data, 300, 30]   # Word embeddings (300D) up to 30 words
train_length:      [num_data]            # Actual length of each sentence
sentence_steps:    int                   # Maximum sentence length (30)
```

### Model Input/Output
```
Text Input:
  - shape: [batch_size, 30, 300]   # (sentence_steps, embedding_dim)
  - type: Word2Vec embeddings (300-dimensional)

Motion Output:
  - shape: [batch_size, 32, 24]    # (action_steps, pose_dimension)
  - content: 24D pose vectors (3D location + 7 joint directions)
             [0:3]    = neck position (x, y, z)
             [3:6]    = joint 1 vector
             [6:9]    = joint 2 vector
             ...
             [21:24]  = joint 7 vector (head)
```

### Pose Interpretation (24D)
- Joint 0: Neck (base/reference point) - BLUE
- Joints 1-2: Shoulders - GREEN  
- Joints 3-4: Elbows - GREEN
- Joints 5-6: Wrists/Hands - GREEN
- Joint 7: Head - RED

---

## Model Architectures

### Seq2Seq Model (seq2seq_structure.py)

**Components:**
1. **Character Encoder LSTM**
   - Input: Text embeddings [batch, sentence_steps, dim_sentence]
   - Output: Text encodings [batch, sentence_steps, dim_char_enc]
   - Parameters: dim_sentence=300, dim_char_enc=300

2. **Char2Action Decoder (Attention-based)**
   - Converts text encodings to motion sequences
   - Uses attention mechanism over text encodings
   - Loop function: uses previous output as next input
   - Output: [batch, action_steps, 24]

3. **Action2Char Decoder**
   - Reconstructs text from motion features
   - For cycle-consistency loss
   - Used only during training

**Key Features:**
- Attention-based decoder with Luong-style attention
- Bidirectional information flow (text → motion → text)
- Supports variable-length sequences with packing/padding
- Loss: MSE(generated_action, real_action) + 5.0 * MSE(reconstructed_text, input_text)

### GAN Model (gan_structure.py)

**Generator:**
- Reuses Seq2Seq encoder and char2action decoder
- Adds random noise to increase diversity
- No cycle-consistency requirement during GAN phase

**Discriminator:**
- Attention-based decoder analyzing motion quality
- Takes text encoding and motion sequence
- Outputs binary classification (real/fake)
- No random noise processing

**Loss Functions:**
```
Discriminator Loss: -mean(log(real_label) + log(1 - fake_label))
Generator Loss:    -mean(log(fake_label))
```

**Training Strategy:**
- Load pretrained Seq2Seq weights for generator initialization
- Separate optimizers for generator and discriminator
- char_encoder frozen during GAN training (no_grad wrapper)
- Higher discriminator learning rate complexity

---

## Main Entry Points & Usage

### 1. Data Preprocessing (utils/process_data.py)
```bash
# Process MSR-VTT dataset into train-ready format
python -m utils.process_data \
    --data_dir ./data \
    --output ./data/metadata.npz \
    --verify
```

**Requirements:**
- `data/GoogleNews-vectors-negative300.bin` (Word2Vec model)
- `data/total_script.txt` (all text descriptions)
- `data/pose/` directory with .mat files
- `data/script/` directory with text files

### 2. Seq2Seq Training (seq2seq_pretrain.py)
```bash
python seq2seq_pretrain.py
```

**Configuration (in seq2seq_pretrain.py):**
```python
dim_sentence = 300          # Word embedding dimension
dim_char_enc = 300          # Encoder hidden size
dim_gen = 300               # Decoder hidden size
batch_size = 32
dim_random = 10             # Random noise dimension
action_steps = 32           # Output motion length
sentence_steps = 30         # Max input text length
max_epoch = 500
learning_rate = 0.00005
save_stride = 5             # Save checkpoint every 5 epochs
```

**Output:** Saves checkpoints to `./seq2seq_model/model_epoch_*.pth`

### 3. GAN Training (gan_train.py)
```bash
python gan_train.py
```

**Configuration:**
```python
seq2seq_model_dir = './seq2seq_model/model_epoch_500.pth'  # Pretrained weights
gen_learning_rate = 0.000002   # 2e-6
dis_learning_rate = 0.000002   # 2e-6
max_epoch = 500
restore = 0                     # Set to 1 to resume training
```

**Process:**
1. Loads pretrained Seq2Seq generator weights
2. Initializes random discriminator weights
3. Alternates generator and discriminator training
4. Saves checkpoints to `./gan_model/model_epoch_*.pth`

### 4. Inference - Simple Testing (test_seq2seq_simple.py)
```bash
python test_seq2seq_simple.py
```

**Generates:**
- Motion sequences from 5 predefined test sentences
- Saves .npy files and visualization PNGs
- Uses zero noise (deterministic output)

### 5. Inference - Enhanced Testing (test_seq2seq_enhanced.py)
```bash
python test_seq2seq_enhanced.py
```

**Generates:**
- Complete motion visualizations with full sentence names
- Multiple output formats:
  - `*_8frames.png` - 8 key frames in grid layout
  - `*_animation.gif` - 32-frame animation
  - `*_32frames_grid.png` - All 32 frames in grid
  - `*_action.npy` - Raw motion data

---

## Code Structure & Key Classes

### Seq2SeqModel
```python
class Seq2SeqModel(nn.Module):
    def char_encoder(x, seq_len) → [batch, sentence_steps, dim_char_enc]
    def char2action(char_enc, init_input, random_noise, batch_size) → actions, features
    def action2char(action_enc, init_input, random_noise, batch_size) → reconstructed_text
    def seq2seq_loss(fake_action, real_action, fake_char, real_char) → total_loss
```

### Seq2SeqTrainer
```python
class Seq2SeqTrainer:
    def train():  # Main training loop
        # For each epoch:
        #   1. Shuffle data
        #   2. For each batch:
        #      - Encode text
        #      - Generate motion from text
        #      - Reconstruct text from motion
        #      - Compute bidirectional loss
        #      - Backprop and update weights
        #   3. Save checkpoint every save_stride epochs
```

### GANModel
```python
class GANModel(nn.Module):
    # Inherits encoder from Seq2Seq
    def char2action(...) → fake_actions_list
    def discriminator(char_seq, action_seq, batch_size) → [batch, 1]
    def dis_loss(real_label, fake_label) → scalar
    def gen_loss(fake_label) → scalar
```

### GANTrainer
```python
class GANTrainer:
    def load_seq2seq_pretrained(seq2seq_path)  # Initialize generator
    def train():  # Alternating GAN training
        # For each epoch and batch:
        #   1. Encode text
        #   2. Generate fake motion
        #   3. Train discriminator on real/fake
        #   4. Train generator to fool discriminator
        #   5. Save checkpoint
```

### Visualization
```python
# seq2seq_enhanced_visualization.py
visualize_key_frames(action_seq, title, save_path)      # 8 frames PNG
visualize_full_animation(action_seq, title, save_path)  # 32-frame GIF
visualize_all_frames_grid(action_seq, title, save_path) # 32-frame grid PNG
visualize_action_complete(action_seq, sentence, output_dir) → file_dict
```

---

## Important Implementation Details

### Attention Mechanism (AttentionDecoder)
```python
# Luong-style attention
scores = tanh(W_a(hidden) + U_a(encoder_outputs))
weights = softmax(scores)
context = sum(weights * encoder_outputs)
lstm_input = concat(decoder_input, context)
```

### Loop Function (Seq2Seq & GAN)
```python
# Used during decoding to use previous outputs as inputs:
if i == 0:
    inp = provided_input
else:
    inp = loop_function(previous_output, i)

# For char2action:
#   action = W_out(decoder_hidden)
#   next_input = W_in(action)
```

### Data Transposition Conventions
```python
# Numpy data layout (from preprocessing):
train_script: [num_data, dim_sentence, sentence_steps]  
train_action: [num_data, dim_action, action_steps]

# PyTorch model input (batch_first=True):
script_batch: [batch, sentence_steps, dim_sentence]
action_batch: [batch, action_steps, dim_action]

# Conversion in training loop:
script_batch = train_script_tensor[indices].transpose(1, 2)
```

### Random Noise Usage
```python
# Seq2Seq training: zero noise (deterministic baseline)
curr_random_c2a = torch.zeros(batch_size, sentence_steps, dim_random)

# GAN training: Gaussian noise (for diversity)
curr_random = torch.randn(batch_size, sentence_steps, dim_random)

# Inference: zero noise (reproducible results)
curr_random = torch.zeros(1, sentence_steps, dim_random)
```

### Model Initialization
```python
# All weights initialized with normal distribution std=0.01
# Matches original TensorFlow implementation
nn.init.normal_(weight, std=0.01)
```

---

## Development Patterns & Conventions

### Parameter Naming
- `dim_sentence`: Text embedding dimension (typically 300)
- `dim_char_enc`: Text encoder hidden size
- `dim_gen`: Generator/decoder hidden size
- `dim_dis`: Discriminator hidden size
- `dim_random`: Random noise dimension
- `sentence_steps`: Maximum text sequence length
- `action_steps`: Fixed motion sequence length

### File Organization
- Model architecture: `*_structure.py`
- Training logic: `*_trainer.py`
- Inference logic: `*_tester.py`
- Visualization: `*_visualization.py`
- Entry points: Root level `*.py` scripts

### Checkpoint Format (PyTorch)
```python
checkpoint = {
    'epoch': epoch_num,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_value
}
torch.save(checkpoint, path)
torch.load(checkpoint, map_location=device)
```

### Device Handling
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set before importing torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
tensor = tensor.to(device)
```

---

## Dependencies & Requirements

**Core Libraries:**
- `torch` (PyTorch) - Deep learning framework
- `numpy` - Numerical computing
- `scipy` - Scientific computing (mat file I/O)
- `gensim` - Word2Vec loading
- `matplotlib` - Visualization and animation
- `tqdm` - Progress bars

**Python Version:** 3.6+

**GPU:** NVIDIA GPU with CUDA support recommended (can run on CPU)

---

## Common Tasks for AI Assistants

### 1. Bug Fixes & Debugging
- Check tensor shape mismatches (especially after transpose operations)
- Verify data loading (numpy vs. torch, transpose conventions)
- Check gradient flow (use `no_grad()` appropriately)
- Verify model parameter initialization

### 2. Feature Enhancements
- Add new loss functions (combine MSE with perceptual losses)
- Implement learning rate scheduling
- Add batch normalization or layer normalization
- Support multi-GPU training

### 3. Inference Scripts
- Create custom test sentences
- Generate motion for arbitrary text
- Export motion to standard formats (BVH, FBX)
- Implement real-time inference

### 4. Visualization Improvements
- Add skeleton type options (different joint structures)
- Implement different camera angles
- Add motion smoothing/interpolation
- Export to video formats

### 5. Data Processing
- Support new dataset formats
- Implement data augmentation
- Add normalization statistics
- Handle missing or corrupted data

---

## Testing & Validation

### Unit Testing Approach
```python
# In model structures (*_structure.py), __main__ sections test:
batch_size = 4
sentence_steps = 30
action_steps = 32

# Create test tensors
script = torch.randn(batch_size, sentence_steps, dim_sentence)
action = torch.randn(batch_size, action_steps, 24)

# Forward pass and check shapes
output = model(script)
assert output.shape == (batch_size, action_steps, 24)
```

### Integration Testing
- Run trainers with small datasets (10 epochs, 100 samples)
- Verify loss decreases over time
- Check checkpoint saving/loading
- Validate inference pipeline

---

## Performance Considerations

### Training
- **Batch Size:** 32 (configurable, larger = faster but needs more GPU memory)
- **Epochs:** 500 recommended for good convergence
- **Time per epoch:** ~10-30 seconds (depending on hardware)
- **Total training time:** ~2-3 hours on modern GPU

### Inference
- **Speed:** ~1 second per sentence (batch size 1)
- **Memory:** ~2GB GPU memory required for inference

### Optimization Tips
1. Use gradient checkpointing for memory efficiency
2. Implement mixed precision training (fp16)
3. Use DataLoader with multiple workers
4. Profile with `torch.profiler` for bottlenecks

---

## Future Improvements

1. **Multi-headed attention** - Better context modeling
2. **Transformer-based architecture** - Replace LSTM entirely
3. **Motion quality metrics** - Perceptual losses, motion smoothness
4. **Controllable generation** - Add style/emotion controls
5. **Multi-modal learning** - Combine video, skeleton, and text
6. **Real-time performance** - Optimize for live inference
7. **Diverse generation** - Better GAN formulation (Wasserstein, etc.)

---

## References & Related Concepts

- **Seq2Seq:** Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014)
- **Attention Mechanism:** Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- **GANs:** Goodfellow et al., "Generative Adversarial Nets" (2014)
- **Motion Generation:** Lin et al., "Text2Action: Generalized Actions from Texts" (2018)
- **Word2Vec:** Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)

---

## Troubleshooting Guide

### CUDA Out of Memory
- Reduce batch_size in config
- Reduce dim_* parameters (300 → 256)
- Use mixed precision training
- Clear GPU memory: `torch.cuda.empty_cache()`

### NaN/Inf Losses
- Check data normalization/preprocessing
- Reduce learning rate
- Check for division by zero in loss computation
- Verify gradient clipping is enabled

### Poor Motion Quality
- Train for more epochs
- Check Word2Vec embeddings are loaded correctly
- Verify data preprocessing (check shapes)
- Try GAN training for refinement

### Slow Training
- Use larger batch size (if GPU memory allows)
- Use GPU instead of CPU
- Check for data loading bottleneck
- Profile with PyTorch profiler

---

**Last Updated:** November 2025
**Status:** PyTorch implementation, test version with Seq2Seq + GAN architecture
