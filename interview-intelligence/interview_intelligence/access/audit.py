from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..db.models import AuditLog


def audit(db: Session, action: str, *, actor_user_id: Optional[uuid.UUID] = None, actor_email: str = "",
          target_type: str = "", target_id: Any = "", meta: Optional[Dict[str, Any]] = None) -> None:
    db.add(AuditLog(
        actor_user_id=actor_user_id,
        actor_email=(actor_email or "")[:320],
        action=action,
        target_type=target_type,
        target_id=str(target_id or "")[:128],
        meta=meta or {},
    ))
