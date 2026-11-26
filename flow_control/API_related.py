import asyncio
import logging
from decimal import Decimal, DecimalException, ROUND_FLOOR

import aiohttp
from comfy_api.latest import io

CONVERSION_RATE = 211.0


class CreditBalance_KASKI(io.ComfyNode):
    """Read credits once per execution and pass ANY data through unchanged."""

    _LOGGER = logging.getLogger(__name__)
    _DEFAULT_API_BASE = "https://api.comfy.org"
    _CREDITS_PER_CENT = Decimal("2.11")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CreditBalance_KASKI",
            display_name="Read Credit Balance",
            category="KASKI/flow_control",
            description=(
                "Current available Comfy.org credits as INT, rounded like the UI. "
                "Uses the current ComfyUI login or API key automatically. "
                "Returns 0 when signed out or the balance cannot be read. "
                "ANY data and lists pass through unchanged. Refreshes on every execution."
            ),
            inputs=[
                io.AnyType.Input(
                    "any_in",
                    optional=True,
                    tooltip="Optional: any data to pass through unchanged.",
                ),
            ],
            outputs=[
                io.AnyType.Output("any_out", is_output_list=True),
                io.Int.Output("credits"),
                io.Float.Output("in_dollar", display_name="In Dollar"),
            ],
            hidden=[
                io.Hidden.auth_token_comfy_org,
                io.Hidden.api_key_comfy_org,
            ],
            is_input_list=True,
            # No frontend login requirement: signed-out runs must return 0.
            is_api_node=False,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    async def execute(cls, any_in=None) -> io.NodeOutput:
        credits = await cls._fetch_credits()
        # Receive the entire Comfy list once; preserve its objects and order.
        return io.NodeOutput(
            any_in if any_in is not None else [None],
            credits,
            credits / CONVERSION_RATE,
        )

    @classmethod
    def _balance_url(cls) -> str:
        from comfy.cli_args import args

        base = getattr(args, "comfy_api_base", None) or cls._DEFAULT_API_BASE
        return base.rstrip("/") + "/customers/balance"

    @classmethod
    def _credits_from_balance(cls, payload) -> int:
        if not isinstance(payload, dict):
            raise ValueError("Unexpected balance response")
        if payload.get("effective_balance_authoritative") is False:
            raise ValueError("Available balance is not authoritative")
        currency = payload.get("currency", "usd")
        if not isinstance(currency, str) or currency.lower() != "usd":
            raise ValueError("Unsupported balance currency")

        # Balance *_micros aliases contain USD cents, like *_cents. Usage
        # fields have different units. Prefer the balance after pending costs.
        for field in (
            "effective_balance_cents",
            "effective_balance_micros",
            "amount_cents",
            "amount_micros",
        ):
            value = payload.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError("Invalid balance value")
            cents = Decimal(str(value))
            if not cents.is_finite():
                raise ValueError("Non-finite balance")
            credits = cents * cls._CREDITS_PER_CENT
            # Match the frontend's Math.round, including negative balances.
            return int((credits + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR))
        raise ValueError("Balance field missing")

    @classmethod
    async def _fetch_credits(cls) -> int:
        # V3 provides hidden credentials on the execution's class clone,
        # independently of the list handling of regular inputs.
        auth_token = getattr(cls.hidden, "auth_token_comfy_org", None)
        api_key = getattr(cls.hidden, "api_key_comfy_org", None)
        if auth_token:
            auth = {"Authorization": "Bearer " + auth_token}
        elif api_key:
            auth = {"X-API-KEY": api_key}
        else:
            return 0

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.get(
                    cls._balance_url(),
                    headers={"Accept": "application/json", "Cache-Control": "no-cache", **auth},
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        cls._LOGGER.warning(
                            "KASKI Credit Balance: balance request returned HTTP %s; returning 0.",
                            response.status,
                        )
                        return 0
                    return cls._credits_from_balance(await response.json(content_type=None))
        except (
            aiohttp.ClientError, asyncio.TimeoutError, OSError,
            ValueError, TypeError, DecimalException,
        ) as error:
            # Never log credentials, response bodies or exception messages.
            cls._LOGGER.warning(
                "KASKI Credit Balance: balance request failed (%s); returning 0.",
                type(error).__name__,
            )
            return 0


API_RELATED_NODES_LIST = [
    CreditBalance_KASKI,
]
