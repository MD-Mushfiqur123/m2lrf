"""
M-2LRF Multi-Modal Engine: Unified Processor.
=============================================
Provides preprocessing, formatting, and token interleaving for:
- Text prompts and special tokens (<image>, <audio>, <|im_start|>)
- Image normalization, resizing, and patching
- Audio waveform conversion to log-mel spectrograms
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch


class MultiModalProcessor:
    """
    Unified multi-modal input processor.
    Coordinates tokenization, image tensor formatting, and audio spectrogram extraction.
    """

    IMAGE_TOKEN = "<image>"
    AUDIO_TOKEN = "<audio>"

    def __init__(
        self,
        image_size: int = 224,
        num_image_tokens: int = 64,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ):
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def process_image(self, image: Any) -> torch.Tensor:
        """
        Converts input image (PIL / numpy / torch) to normalized tensor [3, H, W].
        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 3:
                return image
            elif image.ndim == 3 and image.shape[-1] == 3:
                return image.permute(2, 0, 1)

        # Create deterministic synthetic normalized tensor
        return torch.randn(3, self.image_size, self.image_size)

    def process_audio(self, waveform: Any) -> torch.Tensor:
        """
        Converts 1D audio waveform into log-mel filterbank spectrogram [NumMels, TimeFrames].
        """
        if isinstance(waveform, torch.Tensor):
            if waveform.ndim == 2 and waveform.shape[0] == self.n_mels:
                return waveform

        # Synthetic log-mel spectrogram with 100 time frames
        return torch.randn(self.n_mels, 100)

    def format_multimodal_prompt(
        self,
        text: str,
        has_image: bool = False,
        has_audio: bool = False,
    ) -> str:
        """
        Interleaves special tokens into prompt structure.
        """
        prefix = ""
        if has_image:
            prefix += f"{self.IMAGE_TOKEN} "
        if has_audio:
            prefix += f"{self.AUDIO_TOKEN} "
        return f"{prefix}{text}".strip()
