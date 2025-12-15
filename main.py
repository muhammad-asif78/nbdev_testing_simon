from fastapi import FastAPI
from routers.spatial import router as spatial_router

api = FastAPI()


@api.on_event("startup")
def _startup_load_models():
	try:
		from pipeline.angle_detection.angle_predictor import _load_angle_models

		_load_angle_models()
	except Exception:
		pass

api.include_router(spatial_router, prefix="/api", tags=["spatial"])

