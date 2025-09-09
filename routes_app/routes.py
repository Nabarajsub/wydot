
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from typing import Optional
router = APIRouter()
from controller.query_processor import query_processor


@router.get("/response")
async def response(query: Optional[str] = None):
    result = await query_processor(query=query)
    # return JSONResponse(content=result)
    return result