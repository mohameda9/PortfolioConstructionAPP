from fastapi import FastAPI
from app.routes import PortfolioBuilder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:8080", "http://localhost:3000", "http://localhost:3000/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(PortfolioBuilder.router)
