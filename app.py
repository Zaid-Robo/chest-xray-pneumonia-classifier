"""
app.py – Gradio web interface for Chest X-Ray inference.

Run locally:
    python app.py

Deploy to HuggingFace Spaces:
    Push this file + src/ + configs/ + checkpoints/best_mobilenet_v2.pth
    Set SDK to 'gradio' in the Space settings.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load predictor once at startup ──────────────────────────────────────────
CONFIG_PATH     = "configs/config.yaml"
CHECKPOINT_PATH = "checkpoints/best_mobilenet_v2.pth"

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        from inference import ChestXRayPredictor
        predictor = ChestXRayPredictor(CONFIG_PATH, CHECKPOINT_PATH)
    return predictor


# ─── Prediction function ──────────────────────────────────────────────────────

def classify(image: Image.Image):
    """
    Gradio-compatible predict function.

    Args:
        image: PIL Image from the Gradio upload widget.

    Returns:
        (markdown_text, label_confidence_dict)
    """
    if image is None:
        return "⬆️ Please upload a chest X-ray image.", {}

    try:
        p      = get_predictor()
        result = p.predict(image)
        label  = result["label"]
        conf   = result["confidence"]
        probs  = result["probabilities"]

        icon    = "🟢" if label == "NORMAL" else "🔴"
        summary = (
            f"### {icon} Prediction: **{label}**\n\n"
            f"**Confidence:** `{conf:.1%}`\n\n"
            f"| Class | Probability |\n|---|---|\n"
            + "\n".join(f"| {k} | `{v:.1%}` |" for k, v in probs.items())
        )
        return summary, probs

    except FileNotFoundError as exc:
        msg = f"❌ **Model not loaded.**\n\n`{exc}`"
        return msg, {}
    except Exception as exc:
        logger.exception("Inference error")
        return f"❌ Inference error: `{exc}`", {}


# ─── Gradio Interface ─────────────────────────────────────────────────────────

CSS = """
.gradio-container { font-family: 'Inter', sans-serif; }
footer { display: none !important; }
"""

with gr.Blocks(title="Chest X-Ray Classifier", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown(
        """
        # 🫁 Chest X-Ray Pneumonia Classifier
        Upload a chest X-ray image and the model will classify it as **NORMAL** or **PNEUMONIA**.

        > **Architecture**: Fine-tuned MobileNetV2 · **Dataset**: [keremberke/chest-xray-classification](https://huggingface.co/datasets/keremberke/chest-xray-classification)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", label="📤 Upload Chest X-Ray", height=320, image_mode="RGB"
            )
            classify_btn = gr.Button("🔍 Classify", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_md = gr.Markdown(label="Result", value="*Upload an image and click Classify.*")
            label_out = gr.Label(num_top_classes=2, label="Class Probabilities")

    gr.Markdown(
        """
        ---
        ### ⚠️ Medical Disclaimer
        This tool is for **educational and demonstration purposes only**.
        It is **not** a certified medical device and must not be used for clinical decisions.

        ### 🧠 Model Details
        | Property | Value |
        |---|---|
        | Architecture | MobileNetV2 (ImageNet pretrained, fine-tuned) |
        | Dataset | ~5,800 chest X-rays (NORMAL / PNEUMONIA) |
        | Input size | 224 × 224 px |
        | Training | Weighted CE loss + gradual backbone unfreeze |

        ### 🚀 Scaling Notes
        Given more compute, the next steps would be:
        - Replace MobileNetV2 with EfficientNet-B3 or ResNet-50
        - Multi-fold cross-validation
        - Grad-CAM saliency maps for interpretability
        """
    )

    classify_btn.click(fn=classify, inputs=image_input, outputs=[result_md, label_out])
    image_input.change(fn=classify, inputs=image_input, outputs=[result_md, label_out])

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
