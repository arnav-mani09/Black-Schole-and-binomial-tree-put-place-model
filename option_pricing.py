import math
from typing import Callable, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np


OptionType = Literal["call", "put"]

def intrinsic_value(spot_price: float, strike_price: float, option_type: OptionType) -> float:
    """
    Calculate the intrinsic value of an option.

    :param spot_price: Current price of the underlying asset
    :param strike_price: Strike price of the option
    :param option_type: Type of the option ('call' or 'put')
    :return: Intrinsic value of the option
    """
    if option_type == "call":
        return max(0.0, spot_price - strike_price)
    elif option_type == "put":
        return max(0.0, strike_price - spot_price)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

def norm_cdf(x: float) -> float:
    """Cumulative distribution for a standard normal variable."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_price(spot_price: float, strike_price: float, time_to_maturity: float,
                         risk_free_rate: float, volatility: float, option_type: OptionType) -> float:
    """
    Calculate the Black-Scholes price of a European option.
    """
    d1 = (math.log(spot_price / strike_price) +
          (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity) 

    if option_type == 'call':
        price = (spot_price * norm_cdf(d1) -
                 strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2))
        
    elif option_type == 'put':
        price = (strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2) -
                 spot_price * norm_cdf(-d1))
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    return price

def plot_price_heatmap(
    pricer: Callable[..., float],
    strike_price: float,
    time_to_maturity: float,
    risk_free_rate: float,
    option_type: OptionType,
    spots: Tuple[float, float],
    vols: Tuple[float, float],
    grid: int,
    steps: int | None = None,
    output: str = "option_heatmap.png",
) -> str:
    """
    Generate a heatmap of option prices over spot and volatility ranges.

    :param pricer: pricing function to use (black_scholes_price or binomial_tree_price)
    :param spots: (min_spot, max_spot)
    :param vols: (min_vol, max_vol)
    :param grid: number of points along each axis
    :param steps: binomial steps if you pass binomial_tree_price
    :param output: file to save the heatmap image
    :return: path to the saved image
    """
    spot_vals = np.linspace(spots[0], spots[1], grid)
    vol_vals = np.linspace(vols[0], vols[1], grid)
    surface = np.zeros((grid, grid))

    for i, vol in enumerate(vol_vals):
        for j, spot in enumerate(spot_vals):
            kwargs = dict(
                spot_price=spot,
                strike_price=strike_price,
                time_to_maturity=time_to_maturity,
                risk_free_rate=risk_free_rate,
                volatility=vol,
                option_type=option_type,
            )
            if steps is not None:
                kwargs["steps"] = steps
            surface[i, j] = pricer(**kwargs)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[spot_vals[0], spot_vals[-1], vol_vals[0], vol_vals[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Spot price (S)")
    ax.set_ylabel("Volatility (sigma)")
    ax.set_title(f"{pricer.__name__} {option_type} price heatmap")
    fig.colorbar(mesh, ax=ax, label="Option price")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def binomial_tree_price(spot_price: float, strike_price: float, time_to_maturity: float,
                        risk_free_rate: float, volatility: float, steps: int,
                        option_type: OptionType) -> float:
    """
    Calculate the price of a European option using the Binomial Tree model.
    """
    dt = time_to_maturity / steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(risk_free_rate * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    asset_prices = [spot_price * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    
    # Initialize option values at maturity
    option_values = [intrinsic_value(price, strike_price, option_type) for price in asset_prices]

    # Backward induction to get option price at time 0
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = (p * option_values[j + 1] + (1 - p) * option_values[j]) * math.exp(-risk_free_rate * dt)

    return option_values[0]

# Example usage:
if __name__ == "__main__":
    S = 100  # Spot price
    K = 100  # Strike price
    T = 1    # Time to maturity in years
    r = 0.05 # Risk-free interest rate
    sigma = 0.2 # Volatility

    call_price_bs = black_scholes_price(S, K, T, r, sigma, "call")
    put_price_bs = black_scholes_price(S, K, T, r, sigma, "put")

    call_price_bt = binomial_tree_price(S, K, T, r, sigma, 100, "call")
    put_price_bt = binomial_tree_price(S, K, T, r, sigma, 100, "put")

    print(f"Black-Scholes Call Price: {call_price_bs}")
    print(f"Black-Scholes Put Price: {put_price_bs}")
    print(f"Binomial Tree Call Price: {call_price_bt}")
    print(f"Binomial Tree Put Price: {put_price_bt}")

    heatmap_file = plot_price_heatmap(
        pricer=black_scholes_price,
        strike_price=K,
        time_to_maturity=T,
        risk_free_rate=r,
        option_type="call",
        spots=(60, 140),
        vols=(0.05, 0.6),
        grid=40,
    )
    print(f"Saved heatmap to {heatmap_file}")
