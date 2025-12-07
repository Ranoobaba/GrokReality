#!/usr/bin/env python3
"""
FastAPI web app for Voice Chat with Grok LLM and Podcast Interruption
"""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import UPLOAD_FOLDER
from routes.api_routes import router as api_router
from routes.main_routes import router as main_router

app = FastAPI(
    title="Voice Chat with Grok",
    description="Voice Chat with Grok LLM and Podcast Interruption",
    version="1.0.0",
)

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static files for uploads (if needed)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Include routers
app.include_router(main_router)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
