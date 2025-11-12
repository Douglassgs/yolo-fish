from __future__ import annotations

import logging
from fastapi import FastAPI


def create_app() -> FastAPI:
    """FastAPI application factory.

    Wires routers and sets up startup/shutdown hooks.
    """
    app = FastAPI(title="Fish Detection API", version="0.1.0")

    # Routers
    from .api.health import router as health_router
    from .api.images import router as images_router
    from .api.history import router as history_router
    from .api.stream import router as stream_router

    app.include_router(health_router)
    app.include_router(images_router)
    app.include_router(history_router)
    app.include_router(stream_router)

    @app.on_event("startup")
    async def _on_startup() -> None:
        logging.getLogger("uvicorn").info("Starting Fish Detection API")
        # Initialize database
        try:
            from .repositories.sqlite import init_db

            init_db()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("DB init skipped: %s", exc)

        # Lazy-load model at startup for lower first-request latency.
        try:
            from .services.inference import get_inference_service

            get_inference_service().warmup()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Model warmup skipped: %s", exc)

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        logging.getLogger("uvicorn").info("Shutting down Fish Detection API")

    return app
