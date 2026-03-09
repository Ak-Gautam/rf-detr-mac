# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Model weights abstraction and download system.

Provides a small built-in registry for the Mac-only models retained in this fork.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rfdetr.util.files import _download_file, _validate_file_md5
from rfdetr.util.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class ModelWeightAsset:
    """
    Dataclass representing a model asset with download information.

    This is the standard format for model assets across rf-detr packages.
    Both rf-detr and rf-detr-plus should use this structure for compatibility.

    Attributes:
        filename: The local filename for the model weights
        url: The download URL
        md5_hash: The expected MD5 hash for integrity validation (None if not available)

    Example:
        >>> asset = ModelWeightAsset(
        ...     filename='rf-detr-base.pth',
        ...     url='https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth',
        ...     md5_hash='b4d3ce46099eaed50626ede388caf979'
        ... )
    """

    filename: str
    url: str
    md5_hash: Optional[str] = None


class ModelWeightsBase(Enum):
    """
    Base class for model weight registries.

    This base class ensures compile-time compatibility between rf-detr and rf-detr-plus.
    Both packages should inherit from this class to ensure they have the same interface.

    Each enum member's value must be a ModelWeightAsset instance.

    Example inheritance:
        >>> from rfdetr.assets import ModelWeightAsset
        >>> class MyModelWeights(ModelWeightsBase):
        ...     MODEL_NAME = ModelWeightAsset(
        ...         "model.pth",
        ...         "https://example.com/model.pth",
        ...         "abc123"
        ...     )

    Example usage:
        >>> from rfdetr.assets.model_weights import ModelWeights
        >>> asset = ModelWeights.from_filename("rf-detr-base.pth")
        >>> asset.filename
        'rf-detr-base.pth'
        >>> asset.url  # doctest: +SKIP
        'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth'
    """

    def __new__(cls, asset: ModelWeightAsset) -> "ModelWeightsBase":
        obj = object.__new__(cls)
        obj._value_ = asset
        return obj

    # Convenience properties to access the underlying ModelWeightAsset attributes
    @property
    def filename(self) -> str:
        """Get the filename from the underlying ModelWeightAsset."""
        return self.value.filename

    @property
    def url(self) -> str:
        """Get the URL from the underlying ModelWeightAsset."""
        return self.value.url

    @property
    def md5_hash(self) -> Optional[str]:
        """Get the MD5 hash from the underlying ModelWeightAsset."""
        return self.value.md5_hash

    @classmethod
    def from_filename(cls, filename: str) -> Optional[ModelWeightAsset]:
        """
        Get ModelWeightAsset by filename.

        Args:
            filename: The model filename (e.g., 'rf-detr-base.pth')

        Returns:
            ModelWeightAsset instance if found, None otherwise

        Example:
            >>> asset = ModelWeights.from_filename('rf-detr-base.pth')
            >>> asset.url
            'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth'
        """
        for member in cls:
            if member.value.filename == filename:
                return member.value  # Return the ModelWeightAsset directly
        return None

    @classmethod
    def get_url(cls, filename: str) -> Optional[str]:
        """
        Get download URL for a model by filename.

        Args:
            filename: The model filename

        Returns:
            URL string if found, None otherwise
        """
        asset = cls.from_filename(filename)
        return asset.url if asset else None

    @classmethod
    def get_md5(cls, filename: str) -> Optional[str]:
        """
        Get expected MD5 hash for a model by filename.

        Args:
            filename: The model filename

        Returns:
            MD5 hash string if available, None otherwise
        """
        asset = cls.from_filename(filename)
        return asset.md5_hash if asset else None

    @classmethod
    def list_models(cls) -> list[str]:
        """
        List all available model filenames.

        Returns:
            List of model filenames
        """
        return [member.value.filename for member in cls]


class ModelWeights(ModelWeightsBase):
    """
    Enumeration of available RF-DETR model assets.

    Each enum member's value is a ModelWeightAsset instance containing:
    - filename: The local filename for the model weights
    - url: The download URL
    - md5_hash: The expected MD5 hash for integrity validation

    Example:
        >>> asset = ModelWeights.RF_DETR_BASE
        >>> asset.filename
        'rf-detr-base.pth'
        >>> asset.url
        'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth'
    """

    RF_DETR_MEDIUM = ModelWeightAsset(
        "rf-detr-medium.pth",
        "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
        "7223f764a87b863f02eb8d52bf0ce2ee",
    )
    RF_DETR_LARGE_2026 = ModelWeightAsset(
        "rf-detr-large-2026.pth",
        "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "5cb72153541cbcb9aa6efa26222acc75",
    )

    # Segmentation Models
    RF_DETR_SEG_MEDIUM = ModelWeightAsset(
        "rf-detr-seg-medium.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
        "a49af1562c3719227ad43d0ca53b4c7a",
    )
    RF_DETR_SEG_LARGE = ModelWeightAsset(
        "rf-detr-seg-large.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "275f7b094909544ed2841c94a677d07e",
    )


def download_pretrain_weights(
    pretrain_weights: str,
    redownload: bool = False,
    validate_md5: bool = True,
) -> None:
    """
    Download pretrained weights with optional MD5 validation.

    Args:
        pretrain_weights: Name of the pretrained weights file (e.g., 'rf-detr-base.pth')
        redownload: Force re-download even if file exists
        validate_md5: Whether to validate MD5 hash of downloaded file

    Example:
        >>> download_pretrain_weights('rf-detr-base.pth')  # doctest: +SKIP
        Downloading pretrained weights for rf-detr-base.pth
    """
    # Use basename for registry lookup so absolute paths (e.g.
    # "/content/rf-detr-base.pth") match the registered short name.
    model_name = os.path.basename(pretrain_weights)
    asset: Optional[ModelWeightAsset] = ModelWeights.from_filename(model_name)
    if asset is None:
        return

    url = asset.url
    expected_md5 = asset.md5_hash if validate_md5 else None

    # Check if file exists with correct hash
    if os.path.exists(pretrain_weights) and not redownload:
        if expected_md5 and validate_md5:
            if not _validate_file_md5(pretrain_weights, expected_md5):
                logger.warning(f"Existing file {pretrain_weights} has incorrect MD5 hash. Re-downloading...")
            else:
                logger.info(f"File {pretrain_weights} already exists with correct MD5 hash.")
                return
        else:
            return

    logger.info(f"Downloading pretrained weights for {pretrain_weights}")
    _download_file(
        url=url,
        filename=pretrain_weights,
        expected_md5=expected_md5,
    )


def validate_pretrain_weights(pretrain_weights: str, strict: bool = False) -> bool:
    """
    Validate MD5 hash of pretrained weights file.

    Args:
        pretrain_weights: Path to the pretrained weights file
        strict: If True, raise error on validation failure. If False, just warn.

    Returns:
        True if validation passes or no hash is available, False otherwise

    Raises:
        ValueError: If strict=True and validation fails
        FileNotFoundError: If strict=True and file doesn't exist
    """
    if not os.path.exists(pretrain_weights):
        if strict:
            raise FileNotFoundError(f"Pretrained weights file not found: {pretrain_weights}")
        return False

    # Check if we have a hash for this model
    model_name = os.path.basename(pretrain_weights)
    asset = ModelWeights.from_filename(model_name)

    if asset is None or asset.md5_hash is None:
        # No hash available for validation
        logger.debug(f"No MD5 hash available for {model_name}, skipping validation")
        return True

    if not _validate_file_md5(pretrain_weights, asset.md5_hash):
        error_msg = (
            f"MD5 hash validation failed for {pretrain_weights}. "
            f"The file may be corrupted or tampered with. "
            f"Consider re-downloading with download_pretrain_weights('{model_name}', redownload=True)"
        )
        if strict:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)
        return False

    logger.debug(f"MD5 validation passed for {pretrain_weights}")
    return True
