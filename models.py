"""
Pydantic models for the AI Board Scanner API.

This module defines all the data models used in the API including:
- Error response models for standardized error handling
- Health check models for service status monitoring
- Shape and diagram models for spatial data representation
- Response models for API endpoints
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================================
# ERROR RESPONSE MODELS
# ============================================================================


class ErrorDetail(BaseModel):
    """Standard error response structure"""

    success: bool = Field(False, description="Always false for errors")
    status: str = Field(..., description="Error status (e.g., 'error', 'not_ready')")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "status": "error",
                "code": 500,
                "message": "Pipeline execution failed: Unable to process image",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response wrapper"""

    detail: ErrorDetail = Field(..., description="Error details")


# ============================================================================
# HEALTH CHECK MODELS
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response"""

    success: bool = Field(True, description="Service health status")
    status: Literal["ok", "degraded", "error"] = Field(
        ..., description="Service status"
    )
    code: int = Field(200, description="HTTP status code")
    message: str = Field(..., description="Health status message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "status": "ok",
                "code": 200,
                "message": "Service is healthy",
            }
        }
    )


class ReadinessResponse(BaseModel):
    """Readiness check response"""

    success: bool = Field(..., description="Service readiness status")
    status: Literal["ready", "not_ready"] = Field(..., description="Readiness status")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Readiness message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "status": "ready",
                "code": 200,
                "message": "All checks passed - weights, packages, and modules are ready",
            }
        }
    )


# ============================================================================
# SHAPE AND DIAGRAM MODELS
# ============================================================================


class ShapeType(str, Enum):
    """Available shape types in snake_case format"""

    cloud = "cloud"
    diamond = "diamond"
    double_arrow = "double_arrow"
    pentagon = "pentagon"
    racetrack = "racetrack"
    star = "star"
    triangle = "triangle"
    arrow = "arrow"
    circle = "circle"
    rectangle = "rectangle"
    rounded_rectangle = "rounded_rectangle"


class BBox(BaseModel):
    """Bounding box coordinates"""

    x1: int = Field(..., ge=0, description="Left coordinate (min: 0)")
    y1: int = Field(..., ge=0, description="Top coordinate (min: 0)")
    x2: int = Field(..., ge=0, description="Right coordinate (min: 0)")
    y2: int = Field(..., ge=0, description="Bottom coordinate (min: 0)")

    @field_validator("x2")
    @classmethod
    def x2_greater_than_x1(cls, v, info):
        """Validate that x2 is greater than x1 to ensure valid bounding box."""
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be greater than x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_greater_than_y1(cls, v, info):
        """Validate that y2 is greater than y1 to ensure valid bounding box."""
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError("y2 must be greater than y1")
        return v


class TextLabel(BaseModel):
    """A standalone text label not contained within a shape"""

    id: str = Field(..., description="Unique text label identifier (e.g., 'text1')")
    x: int = Field(..., description="X coordinate of text center")
    y: int = Field(..., description="Y coordinate of text center")
    text: str = Field(..., description="Text content")
    bbox: BBox = Field(..., description="Bounding box of the text")
    width: int = Field(..., description="Width of the text in pixels")
    height: int = Field(..., description="Height of the text in pixels")


class SpatialMapRequest(BaseModel):
    """Request model for spatial map processing."""

    file_path: str


class Canvas(BaseModel):
    """Canvas dimensions"""

    width: int = Field(..., description="Canvas width in pixels")
    height: int = Field(..., description="Canvas height in pixels")


class Node(BaseModel):
    """A node represents a shape detected in the image"""

    id: str = Field(
        ..., pattern="^node\\d+$", description="Unique node identifier (e.g., 'node1')"
    )
    x: int = Field(..., ge=0, description="X coordinate of node center")
    y: int = Field(..., ge=0, description="Y coordinate of node center")
    text: str = Field("", description="Text content extracted from the node")
    shape: Literal[
        "cloud",
        "diamond",
        "double_arrow",
        "pentagon",
        "racetrack",
        "star",
        "triangle",
        "arrow",
        "circle",
        "rectangle",
        "rounded_rectangle",
    ] = Field(..., description="Shape type in snake_case format")
    color: str = Field(
        ...,
        pattern="^#[0-9a-fA-F]{6}$",
        description="Color in hex format (e.g., #fafdff)",
    )
    angle: int = Field(
        0, ge=0, le=359, description="Rotation angle in degrees (0-359)"
    )
    width: int = Field(..., gt=0, description="Width of the node in pixels")
    height: int = Field(..., gt=0, description="Height of the node in pixels")
    bbox: Optional[BBox] = Field(None, description="Bounding box of the node")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Detection confidence score (0-1)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "node1",
                "x": 150,
                "y": 200,
                "text": "Process Start",
                "shape": "rectangle",
                "color": "#3498db",
                "angle": 0,
                "width": 100,
                "height": 50,
                "confidence": 0.95,
            }
        }
    )


class Point(BaseModel):
    """A 2D point coordinate"""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class Edge(BaseModel):
    """An edge represents a connection between two nodes"""

    id: str = Field(
        ..., pattern="^edge\\d+$", description="Unique edge identifier (e.g., 'edge1')"
    )
    source: str = Field(..., pattern="^node\\d+$", description="Source node ID")
    target: str = Field(..., pattern="^node\\d+$", description="Target node ID")
    lineStyle: Literal["solid", "dashed", "dotted"] = Field(
        "solid", description="Line style"
    )
    startArrow: bool = Field(
        False, description="Whether the edge has an arrow at the start"
    )
    endArrow: bool = Field(True, description="Whether the edge has an arrow at the end")
    color: str = Field(
        "#333333", pattern="^#[0-9a-fA-F]{6}$", description="Line color in hex format"
    )
    label: Optional[str] = Field(None, description="Edge label or annotation")
    points: Optional[List[Point]] = Field(
        None, description="Path coordinates for the edge"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Detection confidence score (0-1)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "edge1",
                "source": "node1",
                "target": "node2",
                "lineStyle": "solid",
                "startArrow": False,
                "endArrow": True,
                "color": "#333333",
                "confidence": 0.92,
            }
        }
    )


class ProcessingMetadata(BaseModel):
    """Metadata about the processing operation"""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Processing timestamp (UTC)"
    )
    processing_time_ms: Optional[int] = Field(
        None, ge=0, description="Processing time in milliseconds"
    )
    api_version: str = Field("1.0.0", description="API version")
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings during processing"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time_ms": 1250,
                "api_version": "1.0.0",
                "warnings": [],
            }
        }
    )


class SpatialData(BaseModel):
    """Complete spatial mapping data"""

    canvas: Canvas = Field(..., description="Canvas dimensions")
    nodes: List[Node] = Field(
        default_factory=list, description="List of detected nodes/shapes"
    )
    edges: List[Edge] = Field(
        default_factory=list, description="List of detected edges/connections"
    )
    text_labels: List[TextLabel] = Field(
        default_factory=list, description="List of standalone text labels"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "canvas": {"width": 1920, "height": 1080},
                "nodes": [
                    {
                        "id": "node1",
                        "x": 150,
                        "y": 200,
                        "text": "Start",
                        "shape": "circle",
                        "color": "#3498db",
                        "angle": 0,
                        "width": 100,
                        "height": 100,
                    }
                ],
                "edges": [
                    {
                        "id": "edge1",
                        "source": "node1",
                        "target": "node2",
                        "lineStyle": "solid",
                        "startArrow": False,
                        "endArrow": True,
                        "color": "#333333",
                    }
                ],
                "text_labels": [],
            }
        }
    )


class SpatialMapResponse(BaseModel):
    """Response containing the spatial mapping result with metadata"""

    success: bool = Field(True, description="Operation success status")
    data: SpatialData = Field(
        ...,
        description="Spatial mapping data containing canvas, nodes, edges, and text labels",
    )
    metadata: ProcessingMetadata = Field(
        default_factory=ProcessingMetadata, description="Processing metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "canvas": {"width": 1920, "height": 1080},
                    "nodes": [],
                    "edges": [],
                    "text_labels": [],
                },
                "metadata": {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "processing_time_ms": 1250,
                    "api_version": "1.0.0",
                    "warnings": [],
                },
            }
        }
    )
