"""
Convexity Risk in a Fixed Income Book
--------------------------------------
Article illustration — Jason Bell, 2026

This module demonstrates:
  1. Bond pricing from yield (present value of cash flows)
  2. Modified duration  — first-order price sensitivity to yield
  3. Convexity         — second-order correction (the bit duration misses)
  4. P&L approximation — duration-only vs duration + convexity
  5. A simple book-level convexity report across multiple positions

The point: duration tells you which way the bus is going.
Convexity tells you whether the driver has been drinking.
"""

import math
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# 1. Core bond analytics
# ---------------------------------------------------------------------------

@dataclass
class Bond:
    face_value:  float        # e.g. 1_000_000
    coupon_rate: float        # annual, e.g. 0.05 for 5%
    maturity:    int          # years to maturity
    frequency:   int = 2      # coupon payments per year (2 = semi-annual)
    label:       str = ""

    def cash_flows(self) -> List[tuple]:
        """Return list of (time_in_years, cash_flow) tuples."""
        periods   = self.maturity * self.frequency
        coupon    = (self.coupon_rate * self.face_value) / self.frequency
        flows     = []
        for t in range(1, periods + 1):
            cf = coupon + (self.face_value if t == periods else 0)
            flows.append((t / self.frequency, cf))
        return flows

    def price(self, yield_pa: float) -> float:
        """Dirty price: PV of all cash flows discounted at yield_pa."""
        r = yield_pa / self.frequency
        return sum(
            cf / (1 + r) ** (t * self.frequency)
            for t, cf in self.cash_flows()
        )

    def modified_duration(self, yield_pa: float) -> float:
        """
        Modified duration (years): percentage price change per 1% move in yield.
        Derived analytically from the weighted average time of cash flows.
        """
        r   = yield_pa / self.frequency
        pv  = self.price(yield_pa)
        mac = sum(
            t * (cf / (1 + r) ** (t * self.frequency))
            for t, cf in self.cash_flows()
        ) / pv
        return mac / (1 + yield_pa / self.frequency)

    def convexity(self, yield_pa: float) -> float:
        """
        Convexity: the second derivative of price w.r.t. yield, normalised by price.
        Positive convexity is your friend — prices rise more than duration predicts
        when yields fall, and fall less when yields rise.
        """
        r      = yield_pa / self.frequency
        pv     = self.price(yield_pa)
        freq_sq = self.frequency ** 2
        cx = sum(
            (cf / (1 + r) ** (t * self.frequency))
            * t * (t + 1 / self.frequency)
            for t, cf in self.cash_flows()
        ) / (pv * freq_sq)
        return cx

    def price_change_approx(self, yield_pa: float, delta_y: float) -> dict:
        """
        Approximate price change for a parallel yield shift of delta_y.

        Duration-only:          ΔP ≈ -D_mod × ΔY × P
        Duration + Convexity:   ΔP ≈ (-D_mod × ΔY + 0.5 × C × ΔY²) × P

        Returns both so you can see what convexity is adding to the story.
        """
        p   = self.price(yield_pa)
        d   = self.modified_duration(yield_pa)
        cx  = self.convexity(yield_pa)

        dur_only = -d * delta_y * p
        dur_cx   = (-d * delta_y + 0.5 * cx * delta_y ** 2) * p
        error    = dur_cx - dur_only   # convexity correction

        actual   = self.price(yield_pa + delta_y) - p

        return {
            "yield_shift_bps"     : delta_y * 10_000,
            "duration_only_pnl"   : dur_only,
            "duration_cx_pnl"     : dur_cx,
            "convexity_correction": error,
            "actual_pnl"          : actual,
            "residual_error"      : actual - dur_cx,   # should be small
        }


# ---------------------------------------------------------------------------
# 2. Book-level convexity report
# ---------------------------------------------------------------------------

@dataclass
class Position:
    bond:     Bond
    notional: float           # number of bonds (face value units)
    label:    str = ""

    def dv01(self, yield_pa: float) -> float:
        """Dollar value of a 1bp move. The first thing a rates trader asks."""
        p  = self.bond.price(yield_pa)
        md = self.bond.modified_duration(yield_pa)
        return -md * 0.0001 * p * self.notional

    def dollar_convexity(self, yield_pa: float) -> float:
        """Convexity in P&L terms per (100bp)² move. The second question."""
        p  = self.bond.price(yield_pa)
        cx = self.bond.convexity(yield_pa)
        return 0.5 * cx * p * self.notional


def book_report(positions: List[Position], yield_pa: float, shock_bps: int = 100):
    """
    Print a readable convexity risk report across a list of positions.
    A parallel shift of +/- shock_bps is applied to illustrate the asymmetry.
    """
    delta_y = shock_bps / 10_000

    sep   = "=" * 72
    dash  = "-" * 68
    pnl_u_hdr = f"P&L +{shock_bps}bp"
    pnl_d_hdr = f"P&L -{shock_bps}bp"
    header = (
        f"\n{sep}\n"
        f"  FIXED INCOME BOOK — CONVEXITY REPORT\n"
        f"  Base yield: {yield_pa*100:.2f}%   Shock: ±{shock_bps}bps\n"
        f"{sep}\n"
        f"  {'Label':<20} {'DV01':>12} {'$Convexity':>14} "
        f"{pnl_u_hdr:>12} {pnl_d_hdr:>12}\n"
        f"  {dash}"
    )
    print(header)

    total_dv01 = total_cx = total_up = total_dn = 0

    for pos in positions:
        dv01  = pos.dv01(yield_pa)
        dcx   = pos.dollar_convexity(yield_pa)
        pnl_u = pos.bond.price_change_approx(yield_pa,  delta_y)["duration_cx_pnl"] * pos.notional
        pnl_d = pos.bond.price_change_approx(yield_pa, -delta_y)["duration_cx_pnl"] * pos.notional

        total_dv01 += dv01
        total_cx   += dcx
        total_up   += pnl_u
        total_dn   += pnl_d

        lbl = pos.label or pos.bond.label or "Bond"
        print(
            f"  {lbl:<20} "
            f"{dv01:>12,.0f} "
            f"{dcx:>14,.0f} "
            f"{pnl_u:>12,.0f} "
            f"{pnl_d:>12,.0f}"
        )

    print(f"  {'-'*68}")
    print(
        f"  {'BOOK TOTAL':<20} "
        f"{total_dv01:>12,.0f} "
        f"{total_cx:>14,.0f} "
        f"{total_up:>12,.0f} "
        f"{total_dn:>12,.0f}"
    )
    print(f"{'='*72}\n")

    # Long convexity: gain on yield-down > loss on yield-up (same magnitude shift)
    if abs(total_dn) > abs(total_up):
        print("  ✓ Book is net long convexity — you make more when yields fall")
        print("    than you lose when they rise by the same amount.\n")
    else:
        print("  ✗ Book is net short convexity — asymmetry works against you.\n")


# ---------------------------------------------------------------------------
# 3. Illustration
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Three bonds: short, medium, long — the usual suspects on a rates desk
    short_bond  = Bond(face_value=1_000_000, coupon_rate=0.04, maturity=2,  label="2Y  4% Bond")
    medium_bond = Bond(face_value=1_000_000, coupon_rate=0.045, maturity=10, label="10Y 4.5% Bond")
    long_bond   = Bond(face_value=1_000_000, coupon_rate=0.05, maturity=30, label="30Y 5% Bond")

    base_yield = 0.045   # 4.5% flat curve for simplicity

    # --- Single bond deep-dive ---
    print("\n--- 10Y Bond: Price sensitivity at different shock sizes ---\n")
    print(f"  {'Shock (bps)':<14} {'Dur Only':>12} {'Dur+Cx':>12} "
          f"{'Actual':>12} {'Cx Correction':>16}")
    print(f"  {'-'*66}")

    for bps in [-200, -100, -50, -25, 25, 50, 100, 200]:
        res = medium_bond.price_change_approx(base_yield, bps / 10_000)
        print(
            f"  {bps:>+12}      "
            f"{res['duration_only_pnl']:>12,.0f} "
            f"{res['duration_cx_pnl']:>12,.0f} "
            f"{res['actual_pnl']:>12,.0f} "
            f"{res['convexity_correction']:>16,.0f}"
        )

    print("\n  Note: the gap between 'Dur Only' and 'Actual' widens at large shocks.")
    print("  Convexity correction closes most of that gap. The residual is gamma of gamma.\n")

    # --- Book-level report ---
    book = [
        Position(bond=short_bond,  notional=50,  label="2Y  4% Bond"),
        Position(bond=medium_bond, notional=30,  label="10Y 4.5% Bond"),
        Position(bond=long_bond,   notional=10,  label="30Y 5% Bond"),
    ]

    book_report(book, base_yield, shock_bps=100)
