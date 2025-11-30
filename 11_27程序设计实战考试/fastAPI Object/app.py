from fastapi import FastAPI
from analyze_routers import analyze

app = FastAPI()
app.include_router(analyze.analyze_router, prefix="/analyze", tags=["analyze_model"])


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
