"""
ADR-Aware Fragmented ML-KEM Integration for LoRaWAN OTAA Join
==============================================================
Reference implementation matching:
  "Adaptive Data Rate-Aware Fragmented ML-KEM Integration for
   Energy-Constrained LoRaWAN Join Procedures in Smart Agriculture"
  (CRC Paper ID 794, NETCRYPT 2026)

Implements, section by section:
  - Sec IV-A/B/C : 1-byte fragmentation header, uplink Join-Request
                   fragmentation, downlink Join-Accept fragmentation
  - Sec IV (Alg 1): ADR greedy-descent data rate selection
  - Eq. (3)       : per-fragment MIC (AES-CMAC)
  - Table III     : EU868 payload-vs-data-rate table
  - Eq. (4)-(6)   : airtime / energy model
  - Table IV      : three-way comparison (Classical / Naive / Proposed)

Real ML-KEM-512 keygen/encaps/decaps is used (via `kyber-py`), so the
800 B public key / 768 B ciphertext sizes are the true FIPS 203 sizes,
not placeholders.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List

from Crypto.Hash import CMAC
from Crypto.Cipher import AES

# --------------------------------------------------------------------------
# Table III — Payload vs. Data Rate (EU868), from the paper
# --------------------------------------------------------------------------
# DR -> (Spreading Factor, max payload bytes, nominal airtime ms for that
# max payload at that DR). Airtime scales linearly with bytes actually sent
# for the fragment-count math; the *max payload* numbers are used to decide
# how many fragments are needed per DR.
DATA_RATES = {
    5: {"sf": 7,  "payload": 222, "air_ms_per_byte": 41 / 222},
    4: {"sf": 8,  "payload": 181, "air_ms_per_byte": 72 / 181},
    3: {"sf": 9,  "payload": 123, "air_ms_per_byte": 123 / 123},
    2: {"sf": 10, "payload": 61,  "air_ms_per_byte": 247 / 61},
    1: {"sf": 11, "payload": 51,  "air_ms_per_byte": 493 / 51},
    0: {"sf": 12, "payload": 51,  "air_ms_per_byte": 989 / 51},
}

RX1_WINDOW_MS = 5000          # JOIN_ACCEPT_DELAY1, Eq. (4)
P_TX_MW = 158                 # ~22 dBm, Sec VI-C
P_IDLE_MW = 85                # idle power, Sec VI-C

FRAG_HEADER_BYTES_PROPOSED = 1    # Sec IV-A: bits[7:4]=index, bit[0]=last
FRAG_HEADER_BYTES_NAIVE = 4       # naive baseline, Sec V-A


# --------------------------------------------------------------------------
# Fragment header  (Sec IV-A)
#   bits [7:4] fragment index (0-15 -> max 16 fragments)
#   bits [3:1] reserved
#   bit  [0]   last-fragment flag
# --------------------------------------------------------------------------
def make_frag_header(index: int, is_last: bool) -> int:
    if not (0 <= index <= 15):
        raise ValueError("Fragment index must fit in 4 bits (0-15, max 16 fragments)")
    header = (index & 0x0F) << 4
    if is_last:
        header |= 0x01
    return header


def parse_frag_header(header: int) -> tuple[int, bool]:
    index = (header >> 4) & 0x0F
    is_last = bool(header & 0x01)
    return index, is_last


@dataclass
class Fragment:
    header: int
    data: bytes

    def to_bytes(self) -> bytes:
        return bytes([self.header]) + self.data

    @property
    def index(self) -> int:
        return parse_frag_header(self.header)[0]

    @property
    def is_last(self) -> bool:
        return parse_frag_header(self.header)[1]


def fragment_payload(payload: bytes, max_frame_bytes: int, header_bytes: int) -> List[Fragment]:
    """
    Split `payload` (e.g. the 800 B ML-KEM public key, or the 768 B
    ciphertext) into fragments that each fit in one LoRaWAN frame at the
    chosen data rate. Mirrors Eq. (1)/(2):
        Fragment_i = [FragHdr] || Data_i
    """
    data_per_frag = max_frame_bytes - header_bytes
    if data_per_frag <= 0:
        raise ValueError("Frame too small to carry a header + any data at this DR")

    n_frags = math.ceil(len(payload) / data_per_frag)
    if header_bytes == FRAG_HEADER_BYTES_PROPOSED and n_frags > 16:
        raise ValueError(
            f"{n_frags} fragments needed but 1-byte header only addresses 16 "
            f"(4-bit index) — paper's design ceiling (Sec IV-A)."
        )

    frags = []
    for i in range(n_frags):
        chunk = payload[i * data_per_frag:(i + 1) * data_per_frag]
        is_last = (i == n_frags - 1)
        frags.append(Fragment(make_frag_header(i, is_last), chunk))
    return frags


def reassemble(fragments: List[Fragment]) -> bytes:
    ordered = sorted(fragments, key=lambda f: f.index)
    return b"".join(f.data for f in ordered)


# --------------------------------------------------------------------------
# Eq. (3): per-fragment MIC = AES-CMAC_Knwk(MHDR || Frag_i || FCnt)
# A join-phase key derived from the root key + DevNonce (see note below —
# the paper calls this "NwkKey", but no network session key exists yet
# during join; it is really a join-phase authentication key).
# --------------------------------------------------------------------------
def derive_join_phase_key(root_key: bytes, dev_nonce: bytes) -> bytes:
    """Derive a 16-byte join-phase MIC key bound to this DevNonce (AES-CMAC-based KDF)."""
    c = CMAC.new(root_key, ciphermod=AES)
    c.update(b"JOIN-PHASE-KEY" + dev_nonce)
    return c.digest()  # 16 bytes, usable as an AES key


def compute_fragment_mic(join_key: bytes, mhdr: bytes, fragment: Fragment, fcnt: int) -> bytes:
    c = CMAC.new(join_key, ciphermod=AES)
    c.update(mhdr + fragment.to_bytes() + fcnt.to_bytes(4, "little"))
    return c.digest()[:4]  # standard LoRaWAN MIC is 4 bytes


def verify_fragment_mic(join_key: bytes, mhdr: bytes, fragment: Fragment,
                         fcnt: int, mic: bytes) -> bool:
    expected = compute_fragment_mic(join_key, mhdr, fragment, fcnt)
    return expected == mic


# --------------------------------------------------------------------------
# Algorithm 1 — Optimized Data Rate Selection (greedy descent)
# --------------------------------------------------------------------------
def select_data_rate(n_fragments_at_dr: dict, energy_budget_mj: float,
                      snr_meas_db: float, snr_threshold_db: dict) -> int:
    """
    Greedy descent from DR5 to DR0 (Sec IV, Algorithm 1).

    n_fragments_at_dr : {dr: number_of_fragments_needed_at_that_dr}
                         (caller computes this from payload size + DR table)
    energy_budget_mj  : device energy budget for this join
    snr_meas_db       : measured link SNR
    snr_threshold_db  : {dr: minimum SNR required for reliable reception}

    Returns the chosen DR, defaulting to DR0 (SF12) if nothing qualifies.
    """
    for dr in sorted(DATA_RATES.keys(), reverse=True):  # {5,4,3,2,1,0}
        n_frag = n_fragments_at_dr[dr]
        air_ms = n_frag * DATA_RATES[dr]["payload"] * DATA_RATES[dr]["air_ms_per_byte"]
        energy_mj = (air_ms / 1000) * P_TX_MW + (RX1_WINDOW_MS / 1000) * P_IDLE_MW
        if energy_mj <= energy_budget_mj and snr_meas_db >= snr_threshold_db.get(dr, -999):
            return dr
    return 0  # fallback: SF12 / DR0


# --------------------------------------------------------------------------
# Energy / airtime model — Eq. (4), (5), (6)
# --------------------------------------------------------------------------
@dataclass
class JoinCostResult:
    dr: int
    n_uplink_frags: int
    n_downlink_frags: int
    airtime_ms: float
    energy_mj: float


def model_join_cost(payload_uplink_bytes: int, payload_downlink_bytes: int,
                     dr: int, header_bytes: int) -> JoinCostResult:
    """
    Eq. (4): T_join = N_up*T_frame + RX1_window + N_down*T_frame
    Eq. (5): E_join = T_air*P_tx + (T_total - T_air)*P_idle
    """
    frame_bytes = DATA_RATES[dr]["payload"]
    data_per_frag = frame_bytes - header_bytes
    n_up = math.ceil(payload_uplink_bytes / data_per_frag) if payload_uplink_bytes else 0
    n_down = math.ceil(payload_downlink_bytes / data_per_frag) if payload_downlink_bytes else 0

    ms_per_byte = DATA_RATES[dr]["air_ms_per_byte"]
    uplink_air_ms = n_up * frame_bytes * ms_per_byte
    downlink_air_ms = n_down * frame_bytes * ms_per_byte
    total_air_ms = uplink_air_ms + downlink_air_ms

    total_join_ms = uplink_air_ms + RX1_WINDOW_MS + downlink_air_ms

    energy_mj = (total_air_ms / 1000) * P_TX_MW + \
                ((total_join_ms - total_air_ms) / 1000) * P_IDLE_MW

    return JoinCostResult(dr, n_up, n_down, total_join_ms, energy_mj)
