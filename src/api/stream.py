from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi import status

from ..lib.image_io import load_image_from_bytes
from ..lib.config import get_ws_token_map, resolve_client_id_by_token
from ..services.inference import get_inference_service
from ..services.stream_session import get_stream_session_service
from ..services.history import get_history_service


router = APIRouter(tags=["stream"])


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    token_map = get_ws_token_map()
    token = websocket.query_params.get("token") or websocket.headers.get("x-auth-token")
    if token_map:
        client_id = resolve_client_id_by_token(token)
        if client_id is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="未通过身份验证")
            return
    else:
        client_id = websocket.query_params.get("client_id", "anonymous")

    await websocket.accept()
    stream_svc = get_stream_session_service()
    history_svc = get_history_service()
    infer = get_inference_service()
    session = None
    try:
        session = stream_svc.open(client_id=client_id)
        # 通知客户端会话编号
        await websocket.send_text(json.dumps({"type": "session", "session_id": session.session_id, "client_id": client_id}))
        while True:
            frame_bytes = await websocket.receive_bytes()
            try:
                img, w, h = load_image_from_bytes(frame_bytes)
            except Exception as e:  # noqa: BLE001
                if session:
                    session.inc_error()
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                continue
            # 执行模型推理
            inf = infer.predict_image(img)
            session.inc_frame()
            frame_index = session.frame_count
            # 发送整帧预测元数据
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "prediction",
                        "session_id": session.session_id,
                        "client_id": client_id,
                        "frame_index": frame_index,
                        "image_id": inf.image_id,
                        "width": inf.width,
                        "height": inf.height,
                        "model_version": inf.model_version,
                        "latency_ms": inf.latency_ms,
                        "predictions": inf.predictions,
                    }
                )
            )

            # 按检测结果逐条写入并推送
            for pred in inf.predictions:
                bbox = pred.get("bbox", {})
                key = f"{pred.get('label','')}:{round(bbox.get('x', 0.0), 1)}:{round(bbox.get('y', 0.0), 1)}:{round(bbox.get('w', 0.0), 1)}:{round(bbox.get('h', 0.0), 1)}"
                if key in session.seen_hashes:
                    continue
                session.seen_hashes.add(key)
                detection_seq = session.next_detection_seq()
                detection_index = int(pred.get("detection_index", detection_seq))
                stored = history_svc.write_stream_detection(
                    session_id=session.session_id,
                    client_id=client_id,
                    image_id=inf.image_id,
                    frame_index=frame_index,
                    detection_index=detection_index,
                    detection_seq=detection_seq,
                    label=str(pred.get("label", "fish")),
                    confidence=float(pred.get("confidence", 0.0)),
                )
                if stored:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "fish_detection",
                                "session_id": session.session_id,
                                "client_id": client_id,
                                "frame_index": frame_index,
                                "detection_index": detection_index,
                                "detection_seq": detection_seq,
                                "label": pred.get("label", "fish"),
                                "confidence": pred.get("confidence", 0.0),
                                "bbox": pred.get("bbox", {}),
                            }
                        )
                    )

            # 每处理固定帧数抽样写入历史
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
                        session_id=session.session_id,
                    )
                except Exception:
                    # 历史写入失败时不中断会话
                    pass
    except WebSocketDisconnect:
        # 正常断开连接
        if session:
            stream_svc.close(session, status="closed")
    except Exception:
        if session:
            stream_svc.close(session, status="error")
        raise
    finally:
        if session and not session.closed:
            stream_svc.close(session, status="closed")
