# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import urllib.parse as _urllib_parse
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
import base64
import io
from transformers import AutoProcessor, AutoModelForImageTextToText

# Set HuggingFace to offline mode to prevent downloading remote code
# This ensures all model code and weights are loaded from local checkpoints only
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# Workaround: Cog's response serialization calls urllib.parse.unquote_plus on
# filenames and may pass bytes under some environments, which raises a
# TypeError in Python 3.11. Monkeypatch unquote_plus to be bytes-safe.
_orig_unquote_plus = _urllib_parse.unquote_plus
def _bytes_safe_unquote_plus(string, encoding="utf-8", errors="replace"):
    if isinstance(string, (bytes, bytearray)):
        # Perform '+' -> space on bytes, then unquote
        try:
            replaced = string.replace(b"+", b" ")
            # urllib.parse.unquote accepts str; decode first
            return _urllib_parse.unquote(replaced.decode(encoding, errors=errors), encoding=encoding, errors=errors)
        except Exception:
            # Fallback: coerce to str and let original handle
            return _orig_unquote_plus(str(string), encoding=encoding, errors=errors)
    # Regular str path
    return _orig_unquote_plus(string, encoding=encoding, errors=errors)

_urllib_parse.unquote_plus = _bytes_safe_unquote_plus

# Additional guard: monkeypatch cog.types.get_filename to be resilient to bytes
try:
    import cog.types as _cog_types  # type: ignore
    _orig_get_filename = getattr(_cog_types, "get_filename", None)
    if callable(_orig_get_filename):
        def _safe_get_filename(value):
            try:
                return _orig_get_filename(value)
            except TypeError:
                # Fallback: robust basename extraction without urllib
                try:
                    if isinstance(value, (bytes, bytearray)):
                        value = value.decode("utf-8", errors="replace")
                except Exception:
                    pass
                try:
                    return os.path.basename(str(value))
                except Exception:
                    return "file"
        _cog_types.get_filename = _safe_get_filename  # type: ignore[attr-defined]
except Exception:
    # If cog internals change or are unavailable, continue without patch
    pass


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Set model path to local checkpoints
        model_path = "./checkpoints"
        
        # Verify checkpoints exist
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model checkpoints not found at {model_path}. "
                "Please ensure the model weights are downloaded to the checkpoints folder."
            )
        
        print(f"Loading InternVL3.5-30B-A3B from {model_path}...")
        print("Offline mode enabled - using only local model files")
        
        # Load the model using the standard transformers API
        # This is a 30B MoE model with ~3B active parameters
        # trust_remote_code=True is required for custom model architecture,
        # but with TRANSFORMERS_OFFLINE=1, it only loads from local files
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,  # Explicitly enforce local files only
        ).eval().cuda()
        
        # Load processor (handles both image preprocessing and tokenization)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,  # Explicitly enforce local files only
        )
        
        print("Model loaded successfully!")
        print(f"Device: {next(self.model.parameters()).device}")

    def predict(
        self,
        image: str = Input(description="Input image path or data URL (inside container)"),
        prompt: str = Input(
            description="Text prompt/question about the image",
            default="Please describe this image in detail.",
        ),
        additional_image_1: str = Input(
            description="Optional second image for multi-image understanding",
            default="",
        ),
        additional_image_2: str = Input(
            description="Optional third image for multi-image understanding",
            default="",
        ),
        additional_image_3: str = Input(
            description="Optional fourth image for multi-image understanding",
            default="",
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=1024,
            ge=1,
            le=8192,
        ),
        temperature: float = Input(
            description="Temperature for sampling (higher = more random)",
            default=0.2,
            ge=0.0,
            le=2.0,
        ),
        top_p: float = Input(
            description="Top-p sampling parameter",
            default=0.95,
            ge=0.0,
            le=1.0,
        ),
        top_k: int = Input(
            description="Top-k sampling parameter",
            default=50,
            ge=1,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty (1.0 = no penalty)",
            default=1.0,
            ge=1.0,
            le=2.0,
        ),
        num_beams: int = Input(
            description="Number of beams for beam search (1 = no beam search)",
            default=1,
            ge=1,
            le=5,
        ),
        thinking_mode: bool = Input(
            description="Enable thinking mode for step-by-step reasoning",
            default=False,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        
        try:
            # Helper: open image from file path or data URL
            def open_image_from_string(path_or_data: str) -> Image.Image:
                value = path_or_data
                if isinstance(value, (bytes, bytearray)):
                    value = value.decode("utf-8", errors="replace")
                s = str(value)
                if s.startswith("data:"):
                    # data URL format: data:<mime>;base64,<data>
                    try:
                        header, b64data = s.split(",", 1)
                    except ValueError:
                        raise ValueError("Invalid data URL for image input")
                    binary = base64.b64decode(b64data, validate=False)
                    return Image.open(io.BytesIO(binary)).convert("RGB")
                # Treat as filesystem path
                return Image.open(s).convert("RGB")

            # Load images
            images = []
            
            # Load primary image
            img = open_image_from_string(image)
            images.append(img)
            
            # Load additional images if provided
            for additional_img in [additional_image_1, additional_image_2, additional_image_3]:
                if additional_img:
                    img = open_image_from_string(additional_img)
                    images.append(img)
            
            print(f"Processing {len(images)} image(s)...")
            
            # Format the conversation based on number of images
            if len(images) == 1:
                # Single image
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                # Multiple images
                content = []
                for i in range(len(images)):
                    content.append({"type": "image"})
                    content.append({"type": "text", "text": f"Image-{i+1}"})
                content.append({"type": "text", "text": f"\n{prompt}"})
                
                conversation = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            
            # Enable thinking mode if requested
            if thinking_mode:
                # Add system prompt for thinking mode
                R1_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""
                
                conversation.insert(0, {
                    "role": "system",
                    "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}],
                })
                print("Thinking mode enabled - generating step-by-step reasoning...")
            
            # Apply chat template and process inputs
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            # Process the inputs (images + text)
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
            ).to(self.model.device)
            
            # Prepare generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if temperature > 0 else None,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams,
                "do_sample": temperature > 0,
            }
            
            print(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}...")
            
            # Generate the response
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **generation_config,
                )
            
            # Decode the output
            # Get only the newly generated tokens (skip the input prompt)
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            response = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            print(f"Generated response...")
            
            # Return the generated text directly
            return response
                
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
