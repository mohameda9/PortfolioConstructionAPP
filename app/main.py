from fastapi import FastAPI
from app.routes import PortfolioBuilder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/"
]
print(origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(PortfolioBuilder.router)

# TODO: save model to db and make it reusable (app.post("/training-environment")).
# TODO:
