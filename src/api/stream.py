from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..lib.image_io import load_image_from_bytes
from ..services.inference import get_inference_service
from ..services.stream_session import get_stream_session_service
from ..services.history import get_history_service


router = APIRouter(tags=["stream"])


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    stream_svc = get_stream_session_service()
    history_svc = get_history_service()
    infer = get_inference_service()
    session = None
    try:
        session = stream_svc.open()
        # Inform client of session id
        await websocket.send_text(json.dumps({"type": "session", "session_id": session.session_id}))
        while True:
            frame_bytes = await websocket.receive_bytes()
            try:
                img, w, h = load_image_from_bytes(frame_bytes)
            except Exception as e:  # noqa: BLE001
                if session:
                    session.inc_error()
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                continue
            # run inference
            inf = infer.predict_image(img)
            session.inc_frame()
            # send metadata/predictions
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "prediction",
                        "session_id": session.session_id,
                        "image_id": inf.image_id,
                        "width": inf.width,
                        "height": inf.height,
                        "model_version": inf.model_version,
                        "latency_ms": inf.latency_ms,
                        "predictions": inf.predictions,
                    }
                )
            )
            # send annotated frame as separate binary message
            await websocket.send_bytes(inf.annotated_jpeg)

            # sample to history every 10 frames
            if session.should_sample(10):
                try:
                    history_svc.write_upload_record(
                        record_id=inf.image_id,
                        image_ref=f"ws:{session.session_id}:{session.frame_count}",
                        predictions=inf.predictions,
                        model_version=inf.model_version,
                        latency_ms=inf.latency_ms,
                        request_id=None,  # type: ignore[arg-type]
                        width=inf.width,
                        height=inf.height,
                    )
                except Exception:
                    # do not break on history issues
                    pass
    except WebSocketDisconnect:
        # normal close
        if session:
            stream_svc.close(session, status="closed")
    except Exception:
        if session:
            stream_svc.close(session, status="error")
        raise
    finally:
        if session and not session.closed:
            stream_svc.close(session, status="closed")
