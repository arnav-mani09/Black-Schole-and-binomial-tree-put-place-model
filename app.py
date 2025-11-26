from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from flask import Flask, jsonify, render_template, request

from option_pricing import OptionType, black_scholes_price, binomial_tree_price


app = Flask(__name__)


def get_pricer(model: str) -> Callable[..., float]:
    return binomial_tree_price if model == "binomial" else black_scholes_price


@dataclass
class PricingRequest:
    spot: float
    strike: float
    maturity: float
    rate: float
    vol: float
    option_type: OptionType
    model: str
    steps: int

    @classmethod
    def from_payload(cls, payload: Dict) -> "PricingRequest":
        return cls(
            spot=float(payload.get("spot", 100.0)),
            strike=float(payload.get("strike", 100.0)),
            maturity=float(payload.get("maturity", 1.0)),
            rate=float(payload.get("rate", 0.02)),
            vol=float(payload.get("vol", 0.2)),
            option_type=payload.get("option_type", "call"),
            model=payload.get("model", "bs"),
            steps=int(payload.get("steps", 100)),
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/price", methods=["POST"])
def api_price():
    req = PricingRequest.from_payload(request.get_json(force=True))
    kwargs = dict(
        spot_price=req.spot,
        strike_price=req.strike,
        time_to_maturity=req.maturity,
        risk_free_rate=req.rate,
        volatility=req.vol,
        option_type=req.option_type,
    )
    if req.model == "binomial":
        kwargs["steps"] = req.steps
    price = get_pricer(req.model)(**kwargs)
    return jsonify({"price": price})


@app.route("/api/heatmap", methods=["POST"])
def api_heatmap():
    payload = request.get_json(force=True)
    req = PricingRequest.from_payload(payload)
    spot_min, spot_max = payload.get("spot_range", [req.spot * 0.6, req.spot * 1.4])
    vol_min, vol_max = payload.get("vol_range", [0.05, 0.6])
    grid = int(payload.get("grid", 40))

    spot_values = np.linspace(float(spot_min), float(spot_max), grid)
    vol_values = np.linspace(float(vol_min), float(vol_max), grid)

    pricer = get_pricer(req.model)
    surface: List[List[float]] = []

    for vol in vol_values:
        row = []
        for spot in spot_values:
            kwargs = dict(
                spot_price=float(spot),
                strike_price=req.strike,
                time_to_maturity=req.maturity,
                risk_free_rate=req.rate,
                volatility=float(vol),
                option_type=req.option_type,
            )
            if req.model == "binomial":
                kwargs["steps"] = req.steps
            row.append(pricer(**kwargs))
        surface.append(row)

    return jsonify(
        {
            "spots": spot_values.tolist(),
            "vols": vol_values.tolist(),
            "surface": surface,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
