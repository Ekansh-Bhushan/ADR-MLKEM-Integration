# ADR-Aware Fragmented ML-KEM LoRaWAN Join — Reference Implementation

Code companion to *"Adaptive Data Rate-Aware Fragmented ML-KEM Integration
for Energy-Constrained LoRaWAN Join Procedures in Smart Agriculture"*
(CRC Paper ID 794, NETCRYPT 2026).

## What's real vs. simulated

| Component | Status |
|---|---|
| ML-KEM-512 keygen / encaps / decaps | **Real** — via `kyber-py`, a pure-Python FIPS 203 implementation. Sizes are the true 800 B public key / 768 B ciphertext, not placeholders. |
| 1-byte fragmentation header (Sec IV-A) | **Real** — bit-packed exactly as specified: bits[7:4] index, bit[0] last-fragment flag. |
| Fragment MIC (Eq. 3) | **Real AES-CMAC**, computed per fragment over MHDR‖Fragment‖FCnt, bound to a DevNonce-derived join-phase key. Tamper/replay detection is demonstrated and verified to fail correctly. |
| ADR greedy-descent (Algorithm 1) | **Real logic** — tries DR5→DR0, picks the first DR meeting both the energy budget and SNR threshold. |
| Airtime/energy numbers (Table II–IV) | **Modeled, not measured.** This runs on a laptop, not an ESP32 — `benchmark_computation()` times the real crypto operations on this machine so the *relative* cost (KeyGen vs Encaps vs Decaps) is meaningful, but absolute milliseconds won't match Table II. The airtime model uses linear-interpolated per-byte constants from Table III, which is an approximation of LoRa's actual symbol-time formula — good enough to reproduce the paper's *qualitative* result (ADR saves energy vs. a fixed worst-case DR) but not a bit-exact airtime simulator. |
| LoRa radio / LMIC stack / INA219 measurement | **Not implemented.** This is a protocol-logic simulation, not a hardware driver. Porting `protocol.py`'s header/fragmentation/MIC logic to C for actual LMIC + liboqs on an ESP32 is straightforward since the data structures are simple byte arrays. |

## Files

- `protocol.py` — core protocol: fragmentation, MIC, ADR algorithm, energy model
- `simulate_join.py` — runnable demo: full end-to-end join, Table II/IV-style benchmarks, tamper-detection demo

## Run it

```bash
pip install kyber-py pycryptodome
python3 simulate_join.py
```

## Honest note on the 38% figure

During Q&A prep we established that the energy saving comes **mostly from
ADR selecting a faster data rate**, not from the 1-byte vs. 4-byte header
(those two headers give the same fragment count at DR3 — they only diverge
at DR0/DR1, near the 16-fragment ceiling). `reproduce_table_iv()` in
`simulate_join.py` demonstrates this directly: it logs which DR the naive
scheme is pinned to vs. which DR Algorithm 1 actually selects, so you can
see the saving is attributable to the DR change, not the header size.
The exact percentage you get depends on the SNR and energy-budget values
you pass in — it will not reproduce the paper's exact 38% number because
the underlying per-byte airtime model here is an approximation, but the
mechanism and direction of the result match.

## Extending this

- Swap `kyber-py` for `liboqs-python` if you want to bind to the same
  liboqs 0.7.2 reference C implementation the paper used (closer to a
  real ESP32 port).
- To port to the actual ESP32 + LMIC setup from the paper: the `Fragment`
  byte layout and `compute_fragment_mic` logic in `protocol.py` translate
  directly to C structs + mbedTLS/liboqs AES-CMAC calls.
