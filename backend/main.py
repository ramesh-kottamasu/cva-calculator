from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.exposure import router as exposure_router

app = FastAPI(
    title="FX Forward CVA Calculator",
    description="Monte Carlo CVA exposure profile for FX forward trades",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(exposure_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
