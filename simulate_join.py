"""
End-to-end simulation of the proposed ADR-aware fragmented ML-KEM join,
reproducing the paper's Table II and Table IV style comparisons.

Run:  python3 simulate_join.py
"""

import os
import time
import secrets

from kyber_py.ml_kem import ML_KEM_512

from protocol import (
    fragment_payload, reassemble, Fragment,
    derive_join_phase_key, compute_fragment_mic, verify_fragment_mic,
    select_data_rate, model_join_cost,
    DATA_RATES, FRAG_HEADER_BYTES_PROPOSED, FRAG_HEADER_BYTES_NAIVE,
)


def run_full_join(dr: int, header_bytes: int, simulate_loss_pct: float = 0.0):
    """
    Runs one complete join: real ML-KEM-512 keygen -> fragment public key
    (uplink, Sec IV-B) -> server encaps -> fragment ciphertext (downlink,
    Sec IV-C) -> device reassembles, verifies every MIC, decapsulates.
    """
    root_key = secrets.token_bytes(16)       # AppKey/NwkKey, Sec II
    dev_nonce = secrets.token_bytes(2)       # DevNonce, Sec II
    join_key = derive_join_phase_key(root_key, dev_nonce)
    mhdr = b"\x00"
    fcnt = 0

    frame_bytes = DATA_RATES[dr]["payload"]

    # ---- Device: ML-KEM-512 keygen (Sec IV / Table I platform) ----
    ek, dk = ML_KEM_512.keygen()             # ek = 800 B public key

    # ---- Uplink: fragment + MIC the public key (Eq. 1) ----
    up_frags = fragment_payload(ek, frame_bytes, header_bytes)
    up_mics = [compute_fragment_mic(join_key, mhdr, f, fcnt) for f in up_frags]

    # ---- "Over the air" with optional fragment loss ----
    received_up = [(f, m) for f, m in zip(up_frags, up_mics)
                   if secrets.randbelow(100) >= simulate_loss_pct * 100]
    uplink_ok = len(received_up) == len(up_frags)
    if not uplink_ok:
        return {"success": False, "stage": "uplink", "dr": dr,
                "n_up": len(up_frags), "n_down": None}

    # ---- Server: verify every fragment's MIC, reassemble, encaps ----
    for f, m in received_up:
        assert verify_fragment_mic(join_key, mhdr, f, fcnt, m), "MIC failed (tamper/replay)"
    ek_reassembled = reassemble([f for f, _ in received_up])
    assert ek_reassembled == ek

    shared_secret_server, ct = ML_KEM_512.encaps(ek_reassembled)  # ct = 768 B

    # ---- Downlink: fragment + MIC the ciphertext (Eq. 2) ----
    down_frags = fragment_payload(ct, frame_bytes, header_bytes)
    fcnt_down = fcnt + 1
    down_mics = [compute_fragment_mic(join_key, mhdr, f, fcnt_down) for f in down_frags]

    received_down = [(f, m) for f, m in zip(down_frags, down_mics)
                     if secrets.randbelow(100) >= simulate_loss_pct * 100]
    downlink_ok = len(received_down) == len(down_frags)
    if not downlink_ok:
        return {"success": False, "stage": "downlink", "dr": dr,
                "n_up": len(up_frags), "n_down": len(down_frags)}

    # ---- Device: verify, reassemble, decapsulate ----
    for f, m in received_down:
        assert verify_fragment_mic(join_key, mhdr, f, fcnt_down, m), "MIC failed (tamper/replay)"
    ct_reassembled = reassemble([f for f, _ in received_down])
    shared_secret_device = ML_KEM_512.decaps(dk, ct_reassembled)

    assert shared_secret_device == shared_secret_server, "Shared secret mismatch!"

    return {
        "success": True, "dr": dr,
        "n_up": len(up_frags), "n_down": len(down_frags),
        "pubkey_bytes": len(ek), "ciphertext_bytes": len(ct),
        "shared_secret": shared_secret_device,
    }


def benchmark_computation(n_trials: int = 20):
    """Reproduces Table II: ML-KEM-512 KeyGen / Encaps / Decaps timing."""
    t_keygen, t_encaps, t_decaps = [], [], []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        ek, dk = ML_KEM_512.keygen()
        t_keygen.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        shared, ct = ML_KEM_512.encaps(ek)
        t_encaps.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ML_KEM_512.decaps(dk, ct)
        t_decaps.append((time.perf_counter() - t0) * 1000)

    avg = lambda L: sum(L) / len(L)
    print("Table II-style benchmark (this machine, NOT the ESP32 — for")
    print("relative comparison only; absolute ms will differ from Table II):")
    print(f"  KeyGen        : {avg(t_keygen):7.3f} ms  (n={n_trials})")
    print(f"  Encapsulation : {avg(t_encaps):7.3f} ms")
    print(f"  Decapsulation : {avg(t_decaps):7.3f} ms")
    print()


def reproduce_table_iv(snr_meas_db: float = 8.0, energy_budget_mj: float = 1000.0):
    """
    Reproduces the Table IV three-way comparison.

    Key point the first version of this function got WRONG: the 38% saving
    is NOT mainly from the 1-byte vs 4-byte header (at a fixed DR those are
    nearly identical, as explained during Q&A prep). It comes from
    Algorithm 1 (ADR) picking a FASTER data rate than the naive scheme's
    fixed worst-case DR3, when link quality (SNR) allows it. That's what
    this version actually exercises.
    """
    PUBKEY_BYTES = 800
    CIPHERTEXT_BYTES = 768

    # Typical LoRa SF demodulation SNR floors (dB) -- lower SF needs better SNR.
    snr_threshold_db = {5: -7.5, 4: -10.0, 3: -12.5, 2: -15.0, 1: -17.5, 0: -20.0}

    print("Table IV-style comparison (EU868, measured SNR = "
          f"{snr_meas_db} dB):")
    print(f"{'Scheme':<22}{'DR':>4}{'Air(ms)':>10}{'Energy(mJ)':>12}{'vs Classical':>14}")

    # Classical OTAA: no ML-KEM, single small uplink/downlink frame, fixed DR3
    classical = model_join_cost(23, 20, dr=3, header_bytes=0)
    print(f"{'Classical OTAA':<22}{3:>4}{classical.airtime_ms:>10.0f}"
          f"{classical.energy_mj:>12.2f}{'1.00x':>14}")

    # Naive ML-KEM: NO ADR intelligence -> pinned to worst-case DR3 (Sec V-A)
    naive = model_join_cost(PUBKEY_BYTES, CIPHERTEXT_BYTES, dr=3,
                             header_bytes=FRAG_HEADER_BYTES_NAIVE)
    print(f"{'Naive ML-KEM':<22}{3:>4}{naive.airtime_ms:>10.0f}"
          f"{naive.energy_mj:>12.2f}{naive.energy_mj/classical.energy_mj:>13.2f}x")

    # Proposed: Algorithm 1 picks the best DR under the energy budget + SNR
    n_frag_by_dr = {}
    for dr, spec in DATA_RATES.items():
        data_per_frag = spec["payload"] - FRAG_HEADER_BYTES_PROPOSED
        n_up = -(-PUBKEY_BYTES // data_per_frag)
        n_down = -(-CIPHERTEXT_BYTES // data_per_frag)
        n_frag_by_dr[dr] = n_up + n_down  # used only for greedy comparison

    chosen_dr = select_data_rate(n_frag_by_dr, energy_budget_mj,
                                  snr_meas_db, snr_threshold_db)
    proposed = model_join_cost(PUBKEY_BYTES, CIPHERTEXT_BYTES, dr=chosen_dr,
                                header_bytes=FRAG_HEADER_BYTES_PROPOSED)
    print(f"{'Proposed ADR':<22}{chosen_dr:>4}{proposed.airtime_ms:>10.0f}"
          f"{proposed.energy_mj:>12.2f}{proposed.energy_mj/classical.energy_mj:>13.2f}x")

    savings = 100 * (1 - proposed.energy_mj / naive.energy_mj)
    print(f"\nEnergy saved vs. naive fragmentation: {savings:.1f}%")
    print(f"(ADR selected DR{chosen_dr} instead of naive's fixed DR3 -> "
          f"{proposed.n_uplink_frags} uplink + {proposed.n_downlink_frags} "
          f"downlink fragments vs naive's {naive.n_uplink_frags}+{naive.n_downlink_frags})\n")


if __name__ == "__main__":
    print("=" * 70)
    print(" ADR-Aware Fragmented ML-KEM LoRaWAN Join — reference simulation")
    print("=" * 70, "\n")

    benchmark_computation()
    reproduce_table_iv()

    print("Running one full end-to-end join at DR3 with real ML-KEM-512...")
    result = run_full_join(dr=3, header_bytes=FRAG_HEADER_BYTES_PROPOSED)
    if result["success"]:
        print(f"  SUCCESS — {result['n_up']} uplink + {result['n_down']} downlink "
              f"fragments, {result['pubkey_bytes']}B pubkey / "
              f"{result['ciphertext_bytes']}B ciphertext")
        print(f"  Shared secret (device==server): {result['shared_secret'].hex()[:32]}...")
    else:
        print(f"  FAILED at {result['stage']} stage")

    print("\nRunning a join with MIC tamper detection demo...")
    # demonstrate that a forged/replayed fragment fails MIC verification
    root_key = secrets.token_bytes(16)
    dev_nonce = secrets.token_bytes(2)
    join_key = derive_join_phase_key(root_key, dev_nonce)
    ek, dk = ML_KEM_512.keygen()
    frags = fragment_payload(ek, DATA_RATES[3]["payload"], FRAG_HEADER_BYTES_PROPOSED)
    mic = compute_fragment_mic(join_key, b"\x00", frags[0], 0)
    tampered = Fragment(frags[0].header, b"\xff" + frags[0].data[1:])
    ok_genuine = verify_fragment_mic(join_key, b"\x00", frags[0], 0, mic)
    ok_tampered = verify_fragment_mic(join_key, b"\x00", tampered, 0, mic)
    print(f"  Genuine fragment MIC valid   : {ok_genuine}")
    print(f"  Tampered fragment MIC valid  : {ok_tampered}  (must be False)")
