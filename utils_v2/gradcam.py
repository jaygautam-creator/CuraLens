"""
CuraLens v2 - Grad-CAM Explainability Module
=============================================
Produces Gradient-weighted Class Activation Maps (Grad-CAM) to highlight
the image regions most influential to the model's cancer prediction.

Reference:
    Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Supported model types:
    - The CuraLens v2 multi-modal model (models_v2.multimodal_model)
    - Any Keras model whose CNN backbone uses a named convolutional layer

Usage:
    from utils_v2.gradcam import GradCAM

    cam = GradCAM(model)                        # auto-detect last conv layer
    cam = GradCAM(model, layer_name="top_conv") # or specify explicitly
    heatmap = cam.compute_heatmap(image_array, metadata_array)
    overlay = cam.overlay(original_image, heatmap)

Debug mode:
    Set environment variable DEBUG_GRADCAM=1 to print gradient statistics.
    Raw heatmap numpy arrays are saved to /tmp/gradcam_raw_<timestamp>.npy.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
import cv2


# ---------------------------------------------------------------------------
# Default target layer for EfficientNetB0
# (the last convolutional layer before global average pooling)
# ---------------------------------------------------------------------------
DEFAULT_CONV_LAYER = "top_conv"

# Set DEBUG_GRADCAM=1 in environment to enable verbose gradient logging
_DEBUG = os.environ.get("DEBUG_GRADCAM", "0").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Layer detection helper
# ---------------------------------------------------------------------------

def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Dynamically find the name of the last Conv2D (or DepthwiseConv2D)
    layer in a Keras model.  Useful when layer names are unknown or when
    a SavedModel has been restored with generated names.

    Args:
        model: Any compiled Keras Model.

    Returns:
        Layer name (str) of the last convolutional layer found.

    Raises:
        ValueError: If no convolutional layer is found.
    """
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Conv2DTranspose,
    )

    def _recurse(m):
        """Walk layers recursively so nested sub-models (e.g. MobileNetV2)
        are also searched."""
        last = None
        for layer in m.layers:
            if isinstance(layer, conv_types):
                last = layer.name
            # If this layer is itself a Model (sub-model), recurse into it
            if hasattr(layer, 'layers'):
                inner = _recurse(layer)
                if inner is not None:
                    last = inner
        return last

    last_conv = _recurse(model)
    if last_conv is None:
        raise ValueError(
            "No Conv2D / DepthwiseConv2D layer found in model. "
            "Available layers:\n  " + "\n  ".join(l.name for l in model.layers)
        )
    return last_conv


# ---------------------------------------------------------------------------
# Core GradCAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Map generator.

    Args:
        model      : Keras Model instance to explain.
        layer_name : Name of the convolutional layer to use for the CAM.
                     Pass None or omit to auto-detect the last conv layer.
        class_idx  : Output neuron index (0 for binary sigmoid; leave None
                     to auto-detect single sigmoid output).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = DEFAULT_CONV_LAYER,
        class_idx: Optional[int] = None,
    ) -> None:
        self.model    = model
        self.class_idx = class_idx

        # ── Resolve layer name ───────────────────────────────────────────
        available = [l.name for l in model.layers]

        if layer_name is None:
            # Auto-detect (recurses into sub-models)
            layer_name = _find_last_conv_layer(model)
            print(f"[GradCAM] Auto-detected last conv layer: '{layer_name}'")
        elif layer_name not in available:
            # Fallback: try to auto-detect instead of raising immediately
            print(
                f"[GradCAM WARNING] Layer '{layer_name}' not found "
                f"(model may have been saved/restored with different names). "
                f"Attempting auto-detection …"
            )
            layer_name = _find_last_conv_layer(model)
            print(f"[GradCAM] Fell back to: '{layer_name}'")

        self.layer_name = layer_name

        # ── Locate the actual layer object (may be inside a sub-model) ───
        def _find_layer(m, name):
            for layer in m.layers:
                if layer.name == name:
                    return layer
                if hasattr(layer, 'layers'):
                    found = _find_layer(layer, name)
                    if found is not None:
                        return found
            return None

        conv_layer = _find_layer(model, layer_name)
        if conv_layer is None:
            raise ValueError(f"[GradCAM] Could not locate layer '{layer_name}' in model graph.")

        # ── Gradient model ───────────────────────────────────────────────
        # Outputs: (conv_feature_maps, final_prediction)
        self._grad_model = tf.keras.Model(
            inputs  = model.inputs,
            outputs = [conv_layer.output, model.output],
            name    = "gradcam_extractor",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_heatmap(
        self,
        image: np.ndarray,
        metadata: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> Optional[np.ndarray]:
        """
        Compute the Grad-CAM heatmap for a single sample.

        Key fixes over v2 original:
        - Input tensors are cast to tf.constant and explicitly watched so
          gradients flow even when all backbone variables are frozen.
        - Normalization uses ReLU-then-max-only (not min-max) per the
          original Selvaraju et al. formulation:
              heatmap = relu(heatmap) / (max(heatmap) + eps)
        - Returns None if gradients are zero (blank heatmap guard).

        Args:
            image    : Pre-processed image array of shape (1, H, W, 3) or
                       (H, W, 3) — will be expanded if needed.
            metadata : Metadata array of shape (1, N) or (N,) for the
                       multi-modal model. Pass None for image-only models.
            eps      : Small epsilon to avoid division by zero.

        Returns:
            Normalised heatmap as a float32 numpy array in [0, 1] with
            shape (H, W), or None if gradients are zero / NaN.
        """
        # ── Prepare input tensors ────────────────────────────────────────
        # Cast to tf.constant (not np.ndarray) so the tape can watch them.
        img_t = tf.constant(self._ensure_batch(image), dtype=tf.float32)

        if metadata is not None:
            meta_t = tf.constant(self._ensure_batch(metadata), dtype=tf.float32)
            input_tensors = [img_t, meta_t]
        else:
            input_tensors = img_t

        # ── Gradient tape ─────────────────────────────────────────────────
        # Explicitly watch all input tensors so gradients propagate through
        # frozen backbone layers (non-trainable params are NOT auto-watched).
        with tf.GradientTape() as tape:
            if isinstance(input_tensors, list):
                for t in input_tensors:
                    tape.watch(t)
            else:
                tape.watch(input_tensors)

            conv_outputs, predictions = self._grad_model(
                input_tensors, training=False
            )

            # Binary sigmoid: target is the single output neuron
            if self.class_idx is not None:
                target = predictions[:, self.class_idx]
            else:
                target = predictions[:, 0]

        # ── Compute gradients ─────────────────────────────────────────────
        grads = tape.gradient(target, conv_outputs)

        # ── Debug logging ─────────────────────────────────────────────────
        if _DEBUG:
            if grads is not None:
                g_abs = tf.abs(grads)
                print(
                    f"[GradCAM DEBUG] layer='{self.layer_name}' | "
                    f"grad_mean={float(tf.reduce_mean(g_abs)):.8f} | "
                    f"grad_max={float(tf.reduce_max(g_abs)):.8f}"
                )
                # Save raw numpy heatmap for external inspection
                ts = int(time.time())
                np.save(f"/tmp/gradcam_raw_{ts}.npy", grads.numpy())
                print(f"[GradCAM DEBUG] Raw gradient array saved → /tmp/gradcam_raw_{ts}.npy")
            else:
                print("[GradCAM DEBUG] grads is None (tape returned nothing)")

        # ── Guard: zero or NaN gradients → return None ───────────────────
        if grads is None:
            print(
                "[GradCAM WARNING] tape.gradient() returned None. "
                "This usually means no watched variable contributed to the "
                "target. Returning None heatmap."
            )
            return None

        grad_max = float(tf.reduce_max(tf.abs(grads)))
        if grad_max < eps or np.isnan(grad_max):
            print(
                f"[GradCAM WARNING] Gradient magnitude is effectively zero "
                f"(max={grad_max:.2e}). Heatmap would be blank — returning None."
            )
            return None

        # ── Pool gradients and weight feature maps ────────────────────────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

        conv_out = conv_outputs[0]                              # (h, w, C)
        heatmap  = conv_out @ pooled_grads[..., tf.newaxis]     # (h, w, 1)
        heatmap  = tf.squeeze(heatmap)                          # (h, w)

        # ── Normalise: ReLU then divide by max (Selvaraju et al., eq. 2) -
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        heatmap = heatmap.numpy().astype("float32")

        # Final blank guard (all-zero after ReLU can still happen)
        if heatmap.max() < eps:
            print(
                "[GradCAM WARNING] Heatmap is all-zero after ReLU. "
                "Returning None to prevent blank overlay."
            )
            return None

        return heatmap

    def overlay(
        self,
        original_image: np.ndarray,
        heatmap: Optional[np.ndarray],
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Optional[np.ndarray]:
        """
        Superimpose the Grad-CAM heatmap onto the original image.

        Args:
            original_image : uint8 RGB image array of shape (H, W, 3).
            heatmap        : Float32 array in [0, 1] from compute_heatmap(),
                             or None (returns None gracefully).
            alpha          : Blending factor for the heatmap overlay.
            colormap       : OpenCV colormap constant (default COLORMAP_JET).

        Returns:
            uint8 RGB overlay image of the same shape as original_image,
            or None if heatmap is None.
        """
        if heatmap is None:
            return None

        h, w = original_image.shape[:2]

        # Resize heatmap to match the input image resolution
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to 0-255 uint8 and apply colormap
        heatmap_uint8  = np.uint8(255 * heatmap_resized)
        heatmap_color  = cv2.applyColorMap(heatmap_uint8, colormap)

        # OpenCV uses BGR; convert original image if it's RGB
        if original_image.ndim == 3 and original_image.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_image

        # Weighted addition
        overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)

        # Guard: confirm output is non-empty
        if overlay is None or overlay.size == 0:
            print("[GradCAM WARNING] cv2.addWeighted produced empty overlay.")
            return None

        # Return as RGB
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    def save_overlay(
        self,
        original_image: np.ndarray,
        heatmap: Optional[np.ndarray],
        save_path: str,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> None:
        """
        Compute overlay and save to disk.

        Args:
            original_image : uint8 RGB image array.
            heatmap        : Float32 heatmap from compute_heatmap() or None.
            save_path      : Destination file path (e.g. 'gradcam_out.png').
            alpha          : Blending factor.
            colormap       : OpenCV colormap.
        """
        result = self.overlay(
            original_image, heatmap, alpha=alpha, colormap=colormap
        )
        if result is None:
            print(f"[GradCAM WARNING] Overlay is None; skipping save to {save_path}")
            return
        # cv2.imwrite expects BGR
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[CuraLens v2] Grad-CAM overlay saved → {save_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_batch(arr: np.ndarray) -> np.ndarray:
        """
        Add a leading batch dimension only when the array is un-batched.

        Rules:
          ndim == 1  → 1-D metadata (features,)          → (1, features)
          ndim == 2  → already batched metadata (1, f)    → unchanged
          ndim == 3  → un-batched image (H, W, C)         → (1, H, W, C)
          ndim == 4  → already batched image (1, H, W, C) → unchanged
        """
        arr = np.asarray(arr, dtype="float32")
        if arr.ndim in (1, 3):          # un-batched inputs only
            arr = np.expand_dims(arr, axis=0)
        return arr


# ---------------------------------------------------------------------------
# Convenience function (stateless helper)
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    metadata: Optional[np.ndarray] = None,
    layer_name: Optional[str] = DEFAULT_CONV_LAYER,
    save_path: Optional[str] = None,
    alpha: float = 0.4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    One-shot Grad-CAM generation.

    Args:
        model      : Compiled Keras model.
        image      : Pre-processed image array (H, W, 3) or (1, H, W, 3).
        metadata   : Optional metadata array for multi-modal model.
        layer_name : Target convolutional layer name, or None to auto-detect.
        save_path  : If provided, saves the overlay image to this path.
        alpha      : Blending strength for the overlay.

    Returns:
        Tuple of (heatmap, overlay_image) — either element may be None
        if Grad-CAM computation fails (gradient zero / blank).
    """
    cam = GradCAM(model, layer_name=layer_name)

    # Strip batch dim for the original image used in overlay
    orig_img = np.asarray(image, dtype="float32")
    if orig_img.ndim == 4:
        orig_img = orig_img[0]

    # Clip to [0, 255] and cast for overlay
    display_img = np.clip(orig_img * 255.0 if orig_img.max() <= 1.0 else orig_img,
                          0, 255).astype("uint8")

    heatmap = cam.compute_heatmap(image, metadata=metadata)
    overlay = cam.overlay(display_img, heatmap, alpha=alpha)

    if save_path and overlay is not None:
        cam.save_overlay(display_img, heatmap, save_path=save_path, alpha=alpha)

    return heatmap, overlay


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Add project root so we can import models_v2
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models_v2.multimodal_model import build_multimodal_model

    print("[CuraLens v2] Building model for Grad-CAM demo...")
    model = build_multimodal_model()

    dummy_image    = np.random.rand(1, 224, 224, 3).astype("float32")
    dummy_metadata = np.array([[45, 1, 0, 1]], dtype="float32")

    cam     = GradCAM(model)   # auto-detect layer
    heatmap = cam.compute_heatmap(dummy_image, metadata=dummy_metadata)

    if heatmap is None:
        print("[CuraLens v2] Grad-CAM returned None (expected on untrained model).")
    else:
        print(f"Heatmap shape : {heatmap.shape}")
        print(f"Heatmap range : [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print("[CuraLens v2] Grad-CAM sanity check passed.")

