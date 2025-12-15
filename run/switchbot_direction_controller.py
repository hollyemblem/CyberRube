"""
switchbot_direction_controller.py

A small, reusable SwitchBot Cloud API v1.1 client + direction-to-bot controller.

What you get:
- SwitchBotClient: handles v1.1 auth headers + GET/POST requests
- DirectionSwitchController: maps UP/LEFT/RIGHT (or any strings) to specific Bot deviceIds
- A tiny CLI so you can run:  python -m switchbot_direction_controller LEFT

Auth mechanism (v1.1):
- headers must include: Authorization, sign, t, nonce
- sign = base64(HMAC_SHA256(secret, token + t + nonce)).upper()

Primary reference: SwitchBot Open API docs (GitHub).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import requests


class SwitchBotError(RuntimeError):
    """Raised for SwitchBot API / transport errors."""


@dataclass(frozen=True)
class SwitchBotAuth:
    token: str
    secret: str

    @staticmethod
    def from_env(token_var: str = "SWITCHBOT_TOKEN", secret_var: str = "SWITCHBOT_SECRET") -> "SwitchBotAuth":
        token = os.getenv(token_var, "").strip()
        secret = os.getenv(secret_var, "").strip()
        if not token or not secret:
            raise SwitchBotError(
                f"Missing credentials. Set env vars {token_var} and {secret_var} (SwitchBot Open Token + Secret)."
            )
        return SwitchBotAuth(token=token, secret=secret)


class SwitchBotClient:
    """
    Minimal SwitchBot OpenAPI v1.1 client.

    Usage:
        client = SwitchBotClient(SwitchBotAuth.from_env())
        devices = client.get_devices()
        client.press_bot(device_id)
    """

    def __init__(
        self,
        auth: SwitchBotAuth,
        base_url: str = "https://api.switch-bot.com",
        api_version: str = "v1.1",
        timeout_s: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version.strip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def _sign_headers(self) -> Dict[str, str]:
        # SwitchBot wants a 13-digit timestamp in ms.
        t_ms = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex  # any unique string is fine
        msg = f"{self.auth.token}{t_ms}{nonce}".encode("utf-8")
        key = self.auth.secret.encode("utf-8")

        digest = hmac.new(key, msg, hashlib.sha256).digest()
        sign = base64.b64encode(digest).decode("utf-8").upper()

        return {
            "Authorization": self.auth.token,
            "sign": sign,
            "t": t_ms,
            "nonce": nonce,
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.base_url}/{self.api_version}/{path}"

    def _raise_for_switchbot(self, payload: Any, status_code: int) -> None:
        """
        SwitchBot API often returns:
          {"statusCode": 100, "body": ..., "message": "success"}
        """
        if status_code != 200:
            raise SwitchBotError(f"HTTP {status_code}: {payload}")

        if isinstance(payload, dict) and "statusCode" in payload:
            if payload.get("statusCode") != 100:
                raise SwitchBotError(f"SwitchBot statusCode={payload.get('statusCode')}: {payload}")

    def get_devices(self) -> Dict[str, Any]:
        resp = self.session.get(self._url("devices"), headers=self._sign_headers(), timeout=self.timeout_s)
        payload = resp.json()
        self._raise_for_switchbot(payload, resp.status_code)
        return payload

    def send_command(self, device_id: str, command: str, parameter: str = "default", command_type: str = "command") -> Dict[str, Any]:
        body = {"command": command, "parameter": parameter, "commandType": command_type}
        resp = self.session.post(
            self._url(f"devices/{device_id}/commands"),
            headers=self._sign_headers(),
            json=body,
            timeout=self.timeout_s,
        )
        payload = resp.json()
        print(payload)
        self._raise_for_switchbot(payload, resp.status_code)
        return payload

    def press_bot(self, device_id: str) -> Dict[str, Any]:
        # For SwitchBot Bot, the standard action is command="press"
        return self.send_command(device_id=device_id, command="turnOn", parameter="default", command_type="command")


class DirectionSwitchController:
    """
    Maps a direction string (e.g., 'UP', 'LEFT', 'RIGHT') to a SwitchBot Bot deviceId and presses it.

    Provide mapping via dict or environment variables:
        SWITCHBOT_UP_ID, SWITCHBOT_LEFT_ID, SWITCHBOT_RIGHT_ID
    """

    def __init__(self, client: SwitchBotClient, mapping: Mapping[str, str]) -> None:
        self.client = client
        # Normalize keys to uppercase to make inputs flexible (up, Up, UP)
        self.mapping = {k.upper(): v for k, v in dict(mapping).items()}

    @staticmethod
    def mapping_from_env(prefix: str = "SWITCHBOT_") -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for key in ("UP", "LEFT", "RIGHT"):
            v = os.getenv(f"{prefix}{key}_ID", "").strip()
            if v:
                mapping[key] = v
        return mapping

    def trigger(self, direction: str) -> Dict[str, Any]:
        d = direction.strip().upper()
        if d not in self.mapping:
            raise SwitchBotError(f"Unknown direction '{direction}'. Known: {sorted(self.mapping.keys())}")
        device_id = self.mapping[d]
        return self.client.press_bot(device_id)