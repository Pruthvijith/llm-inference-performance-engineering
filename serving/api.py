"""Start the loopback-only demonstration with uvicorn serving.api:create_app --factory."""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import os
from .runtime import authorised, Busy, from_environment


class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=1500)
    tokens: int = Field(default=3, ge=1, le=12)


def create_app(worker=None, secret=None):
    secret = secret or os.getenv("SERVING_SECRET", "")
    if len(secret) < 24:
        raise ValueError("Configure a secret of at least 24 characters")
    worker = worker or from_environment()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/v1/complete")
    def complete(payload: Request, authorization: str = Header(default="")):
        if not authorised(authorization, secret):
            raise HTTPException(401, "Invalid credentials")
        try:
            return worker.infer(payload.prompt, payload.tokens)
        except Busy as exc:
            raise HTTPException(429, "Worker busy") from exc
        except ValueError as exc:
            raise HTTPException(422, "Invalid prompt") from exc

    return app
