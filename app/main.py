from fastapi import FastAPI
from app.routes import PortfolioBuilder

app = FastAPI()

app.include_router(PortfolioBuilder.router)

# TODO: save model to db and make it reusable (app.post("/training-environment")).
# TODO:
