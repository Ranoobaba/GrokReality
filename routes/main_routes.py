"""
Main routes for the FastAPI application.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Setup templates
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with voice chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})
