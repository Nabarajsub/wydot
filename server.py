#this is the server page
from fastapi import FastAPI,Request
from routes_app import routes
import uvicorn
import os
app = FastAPI()

app.include_router(
    router=routes.router,
    prefix='/wyollm',
    responses={404: {'description': 'Not found'}},)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)