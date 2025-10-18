# InternVL3.5-30B-A3B Cog Implementation

This is a [Replicate Cog](https://github.com/replicate/cog) implementation of [OpenGVLab/InternVL3_5-30B-A3B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-HF), a powerful open-source multimodal model.

## Model Overview

InternVL3.5-30B-A3B is a Mixture-of-Experts (MoE) vision-language model with:
- **30.5B total parameters** with **~3B active parameters** during inference
- **Advanced reasoning capabilities** via Cascade Reinforcement Learning
- **Multi-image understanding** - process up to 4 images simultaneously
- **Thinking mode** - step-by-step reasoning for complex tasks
- **High performance** - achieves state-of-the-art results on multiple benchmarks

## Features

- ✅ **Offline mode enabled** - All model files loaded from local checkpoints
- ✅ **8-bit quantization** - Reduced memory usage (~30-40GB VRAM)
- ✅ **Streaming inference** - Real-time response generation
- ✅ **Multi-image support** - Analyze multiple images in one request
- ✅ **Thinking mode** - Enable step-by-step reasoning
- ✅ **Flexible generation** - Configurable temperature, top-p, top-k, and more

## Setup

### 1. Download Model Weights

Download the model weights from HuggingFace and place them in the `checkpoints` folder:

```bash
# Install huggingface-cli if not already installed
pip install -U "huggingface_hub[cli]"

# Download the model (requires ~60GB storage)
huggingface-cli download OpenGVLab/InternVL3_5-30B-A3B-HF --local-dir ./checkpoints
```

### 2. Build the Cog Container

```bash
cog build
```

This will:
- Install all dependencies from `requirements.txt`
- Set up CUDA 12.4 environment
- Prepare the model for inference

### 3. Run Predictions

#### Single Image Example

```bash
cog predict \
  -i image=@example.jpg \
  -i prompt="Please describe this image in detail."
```

#### Multi-Image Example

```bash
cog predict \
  -i image=@image1.jpg \
  -i additional_image_1=@image2.jpg \
  -i prompt="Compare these two images and describe their differences."
```

#### Thinking Mode Example

```bash
cog predict \
  -i image=@math_problem.jpg \
  -i prompt="Solve this math problem step by step." \
  -i thinking_mode=true
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | Path | *required* | Primary input image |
| `prompt` | String | "Please describe this image in detail." | Text prompt/question about the image(s) |
| `additional_image_1` | Path | None | Optional second image |
| `additional_image_2` | Path | None | Optional third image |
| `additional_image_3` | Path | None | Optional fourth image |
| `max_new_tokens` | Integer | 1024 | Maximum tokens to generate (1-8192) |
| `temperature` | Float | 0.2 | Sampling temperature (0.0-2.0) |
| `top_p` | Float | 0.95 | Top-p sampling (0.0-1.0) |
| `top_k` | Integer | 50 | Top-k sampling (≥1) |
| `repetition_penalty` | Float | 1.0 | Repetition penalty (1.0-2.0) |
| `num_beams` | Integer | 1 | Number of beams for beam search (1-5) |
| `thinking_mode` | Boolean | False | Enable step-by-step reasoning |

## Use Cases

### 1. Image Description & Understanding
```bash
cog predict -i image=@photo.jpg -i prompt="Describe this image in detail."
```

### 2. Visual Question Answering
```bash
cog predict -i image=@chart.jpg -i prompt="What is the trend shown in this chart?"
```

### 3. OCR & Document Understanding
```bash
cog predict -i image=@document.jpg -i prompt="Extract all the text from this document."
```

### 4. Mathematical Reasoning
```bash
cog predict -i image=@math.jpg -i prompt="Solve this equation." -i thinking_mode=true
```

### 5. Multi-Image Comparison
```bash
cog predict \
  -i image=@before.jpg \
  -i additional_image_1=@after.jpg \
  -i prompt="What changed between these two images?"
```

### 6. Visual Reasoning
```bash
cog predict \
  -i image=@puzzle.jpg \
  -i prompt="Explain the logic behind this visual puzzle." \
  -i thinking_mode=true \
  -i max_new_tokens=2048
```

## Technical Details

### Offline Mode

This implementation uses **offline mode** to ensure security and reproducibility:

```python
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
```

All model code and weights are loaded from local `./checkpoints` directory only. The `trust_remote_code=True` parameter is still required for the custom model architecture, but combined with `local_files_only=True`, it only loads from local files.

### Memory Optimization

The model uses **8-bit quantization** via `load_in_8bit=True`, which:
- Reduces VRAM usage from ~60GB to ~30-40GB
- Maintains model performance
- Enables deployment on single A100 GPU

### Image Preprocessing

The model uses **dynamic resolution** via the `vision_preprocess` method:
- Automatically adjusts to optimal resolution
- Returns pixel values and patch information
- Supports variable image sizes

## Performance

Based on InternVL3.5 benchmarks:
- **MMMU**: State-of-the-art reasoning on multimodal understanding
- **MathVista**: +16.0% improvement in mathematical reasoning
- **OCRBench**: Excellent OCR and document understanding
- **MMBench**: Top performance on comprehensive multimodal tasks
- **4.05× speedup** compared to InternVL3

## System Requirements

- **GPU**: A100 (40GB or 80GB) recommended
- **VRAM**: ~30-40GB with 8-bit quantization
- **Storage**: ~60GB for model weights
- **CUDA**: 12.4
- **Python**: 3.11

## Troubleshooting

### "Model checkpoints not found" Error
Ensure you've downloaded the model weights to `./checkpoints`:
```bash
huggingface-cli download OpenGVLab/InternVL3_5-30B-A3B-HF --local-dir ./checkpoints
```

### Out of Memory Error
Try reducing `max_new_tokens` or ensure you're using an A100 GPU with sufficient VRAM.

### Slow Inference
- Use lower resolution images
- Reduce `num_beams` to 1 (default)
- Decrease `max_new_tokens`

## License

This project is released under the Apache 2.0 License, consistent with the original InternVL3.5 model.

## Citation

```bibtex
@article{wang2025internvl3_5,
  title={InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency},
  author={Wang, Weiyun and Gao, Zhangwei and Gu, Lixin and Pu, Hengjun and Cui, Long and Wei, Xingguang and Liu, Zhaoyang and Jing, Linglin and Ye, Shenglong and Shao, Jie and others},
  journal={arXiv preprint arXiv:2508.18265},
  year={2025}
}
```

## Resources

- [Model Card](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-HF)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [Paper](https://arxiv.org/abs/2508.18265)
- [Replicate Cog Documentation](https://github.com/replicate/cog)

