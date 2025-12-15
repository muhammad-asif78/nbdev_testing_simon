"""
API router for spatial mapping endpoints.

This module provides endpoints for:
- Health and readiness checks
- Converting diagrams to JSON spatial representations
"""

import importlib
import os
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from models import (
    SpatialMapResponse,
    ProcessingMetadata,
    HealthResponse,
    ReadinessResponse,
    ErrorDetail,
)
from pipeline.spatial_pipeline import run_spatial_mapping


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the service."""
    return HealthResponse(
        success=True, status="ok", code=status.HTTP_200_OK, message="Service is healthy"
    )


@router.get(
    "/ready", response_model=ReadinessResponse, responses={503: {"model": ErrorDetail}}
)
async def readiness_check():
    """Check if the service is ready to handle requests."""
    # Check all required model files and weights
    required_files = {
        "RF-DETR weights": "weights/pre-trained-model/checkpoint_best_regular.pth",
        "Angle model weights": "weights/angle-models/Triangle.pth",
        "SAM2 checkpoint": "sam2_checkpoints/sam2_hiera_base_plus.pt",
    }

    # Check SAM2 config in package installation
    try:
        import sam2  # pylint: disable=import-outside-toplevel

        sam2_config_path = os.path.join(
            os.path.dirname(sam2.__file__), "sam2_hiera_b+.yaml"
        )
        if not os.path.exists(sam2_config_path):
            # Try alternative config name
            sam2_config_path = os.path.join(
                os.path.dirname(sam2.__file__), "sam2_hiera_base_plus.yaml"
            )
        if os.path.exists(sam2_config_path):
            required_files["SAM2 config"] = sam2_config_path
    except ImportError:
        pass

    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name} ({path})")

    if missing_files:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                success=False,
                status="not_ready",
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=f"Missing required files: {', '.join(missing_files)}",
            ).model_dump(),
        )

    # Check all required packages
    required_packages = ["paddleocr", "cv2", "skimage", "fitz"]
    missing_packages = []
    for pkg in required_packages:
        if importlib.util.find_spec(pkg) is None:
            missing_packages.append(pkg)

    if missing_packages:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                success=False,
                status="not_ready",
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=f"Missing required packages: {', '.join(missing_packages)}",
            ).model_dump(),
        )

    # Simple check that main pipeline module exists
    try:
        import pipeline.spatial_pipeline  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                success=False,
                status="not_ready",
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=f"Pipeline module not available: {str(e)}",
            ).model_dump(),
        ) from e

    return ReadinessResponse(
        success=True,
        status="ready",
        code=status.HTTP_200_OK,
        message="All checks passed - weights, packages, and modules are ready",
    )


@router.post(
    "/diagram-to-json",
    response_model=SpatialMapResponse,
    responses={500: {"model": ErrorDetail}, 400: {"model": ErrorDetail}},
)
async def spatial_map(file: UploadFile = File(...)):
    """Convert a diagram image to JSON spatial representation."""
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".pdf", ".bmp", ".tiff"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                success=False,
                status="error",
                code=status.HTTP_400_BAD_REQUEST,
                message=(
                    f"Unsupported file type: {file_ext}. "
                    f"Allowed types: {', '.join(allowed_extensions)}"
                ),
            ).model_dump(),
        )

    # Start timing
    start_time = time.time()

    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    contents = await file.read()

    # Check file size (limit to 50MB)
    max_size_mb = 50
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(
                success=False,
                status="error",
                code=status.HTTP_400_BAD_REQUEST,
                message=f"File size exceeds {max_size_mb}MB limit",
            ).model_dump(),
        )

    with open(temp_filename, "wb") as f:
        f.write(contents)

    try:
        result_json = run_spatial_mapping(temp_filename)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create metadata
        metadata = ProcessingMetadata(
            timestamp=datetime.utcnow(),
            processing_time_ms=processing_time_ms,
            api_version="1.0.0",
            warnings=[],
        )

        # Wrap the result in SpatialMapResponse structure
        return SpatialMapResponse(success=True, data=result_json, metadata=metadata)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                success=False,
                status="error",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"Pipeline execution failed: {str(e)}",
            ).model_dump(),
        ) from e
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass
