from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers.api import router as api_router
import uvicorn

app = FastAPI(title="VQA-CL Backend", description="Backend for VQA Curriculum Learning Framework")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "VQA-CL API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
