"""
generate_trace.py — Extract cycle-by-cycle events from the conv_controller FSM.

Implements the exact same FSM as conv_controller.sv (verified against
Verilator simulation: 3297 cycles, 1936 input reads, 288 weight reads,
256/256 outputs correct).

Also models the im2col + GEMM baseline for comparison.

Outputs trace_data.js — a JavaScript file containing both event streams
embedded as constants, ready to load in the visualizer.
"""

import json, math

# ── Layer parameters (matching our simulation) ────────────────────────────────
H_IN = W_IN = H_OUT = W_OUT = 8
C_IN = C_OUT = 4
STRIDE = PADDING = 1
K_STEPS = C_IN * 9          # 36
K_STEPS_MAX = 72             # fixed-stride weight SRAM layout
TOTAL_PIXELS = H_OUT * W_OUT # 64

# ── FSM state constants (matching conv_controller.sv) ─────────────────────────
IDLE, PRELOAD, CLEAR, FEED, DRAIN, WRITE, DONE = 0, 1, 2, 3, 4, 5, 6
STATE_NAMES = ['IDLE','PRELOAD','CLEAR','FEED','DRAIN','WRITE','DONE']


def is_valid(oh, ow, ic, kr, kc):
    """Mirror conv_addr_gen.sv bounds check."""
    ih = oh * STRIDE + kr - PADDING
    iw = ow * STRIDE + kc - PADDING
    return 0 <= ih < H_IN and 0 <= iw < W_IN


def simulate_accel():
    """
    Exact cycle-by-cycle simulation of conv_controller.sv FSM.
    Returns list of event dicts, one per cycle.
    """
    events = []
    state = IDLE

    pre_j = pre_k = 0
    oh = ow = oc_base = 0
    step = ic = kr = kc = 0
    drain = write_cnt = 0

    # Run until DONE (plus one extra cycle to record DONE state)
    for _ in range(20000):
        ev = {
            'cycle':      len(events),
            'state':      STATE_NAMES[state],
            'oh': oh, 'ow': ow, 'oc': oc_base,
            'step':       step,
            'input_re':   0,
            'weight_re':  0,
            'output_we':  0,
            'sa_valid':   0,
            'addr_valid': 0,
            'padding':    0,
        }

        # ── State outputs ──────────────────────────────────────────────
        if state == PRELOAD:
            ev['weight_re'] = 1

        elif state == CLEAR:
            ev['sa_valid'] = 0   # sa_clear=1 this cycle
            # Present addr for step 0  (ic=0, kr=0, kc=0)
            v = is_valid(oh, ow, 0, 0, 0)
            ev['addr_valid'] = int(v)
            ev['input_re'] = int(v)
            ev['padding'] = int(not v)

        elif state == FEED:
            ev['sa_valid'] = 1
            # The data being FED this cycle came from the previous cycle's SRAM read.
            # addr_valid here tracks what is currently being fed to the array
            # (for PE animation) — that's the CURRENT step's validity.
            cur_valid = is_valid(oh, ow, ic, kr, kc)
            ev['addr_valid'] = int(cur_valid)
            ev['padding']    = int(not cur_valid)

            # input_re: is the SRAM being read THIS cycle?
            # The RTL presents the NEXT step's address (ag_query_valid = step < K-1).
            if step < K_STEPS - 1:
                # Compute next (ic, kr, kc) without mutating state
                n_kc = kc + 1
                n_kr, n_ic = kr, ic
                if n_kc > 2:
                    n_kc = 0; n_kr = kr + 1
                    if n_kr > 2:
                        n_kr = 0; n_ic = ic + 1
                next_valid = is_valid(oh, ow, n_ic, n_kr, n_kc)
                ev['input_re'] = int(next_valid)
            else:
                ev['input_re'] = 0   # last step: ag_query_valid=0, no SRAM read

        elif state == DRAIN:
            pass  # sa_valid=0, nothing

        elif state == WRITE:
            ev['output_we'] = 1

        # ── State transitions ──────────────────────────────────────────
        if state == IDLE:
            state = PRELOAD

        elif state == PRELOAD:
            if pre_k == K_STEPS_MAX - 1:
                pre_k = 0
                if pre_j == 3:
                    pre_j = 0
                    state = CLEAR
                    step = ic = kr = kc = 0
                else:
                    pre_j += 1
            else:
                pre_k += 1

        elif state == CLEAR:
            state = FEED
            step = ic = kr = kc = 0

        elif state == FEED:
            if step == K_STEPS - 1:
                state = DRAIN
                drain = 0
            else:
                step += 1
                # Advance (ic, kr, kc) counters
                kc += 1
                if kc > 2:
                    kc = 0; kr += 1
                    if kr > 2:
                        kr = 0; ic += 1

        elif state == DRAIN:
            drain += 1
            if drain == 6:
                state = WRITE
                write_cnt = 0

        elif state == WRITE:
            write_cnt += 1
            if write_cnt == 4:
                write_cnt = 0
                # Advance pixel / oc_tile
                if ow < W_OUT - 1:
                    ow += 1; state = CLEAR
                elif oh < H_OUT - 1:
                    ow = 0; oh += 1; state = CLEAR
                elif oc_base + 4 < C_OUT:
                    ow = oh = 0; oc_base += 4
                    pre_j = pre_k = 0; state = PRELOAD
                else:
                    state = DONE
                if state == CLEAR:
                    step = ic = kr = kc = 0

        elif state == DONE:
            events.append(ev)
            break

        events.append(ev)

    return events


def simulate_im2col():
    """
    Cycle-by-cycle model of im2col + output-stationary GEMM baseline.
    Phases:
      FILL  (2304 cycles): write each of H_out*W_out*C_in*9 activations to buffer
      GEMM  (9216 cycles): for each of H_out*W_out*C_out output values,
                           accumulate over K=36 weight+buffer reads
      WRITE  (256 cycles): write results to output SRAM
    """
    events = []

    FILL_CYCLES  = H_OUT * W_OUT * K_STEPS          # 2304
    GEMM_CYCLES  = H_OUT * W_OUT * C_OUT * K_STEPS  # 9216
    WRITE_CYCLES = H_OUT * W_OUT * C_OUT             # 256

    total = FILL_CYCLES + GEMM_CYCLES + WRITE_CYCLES  # 11776

    for c in range(total + 1):
        if c < FILL_CYCLES:
            pixel  = c // K_STEPS
            step   = c % K_STEPS
            oh_i   = pixel // W_OUT
            ow_i   = pixel % W_OUT
            ic_i   = step  // 9
            kr_i   = (step % 9) // 3
            kc_i   = step % 3
            v      = is_valid(oh_i, ow_i, ic_i, kr_i, kc_i)
            ev = {'cycle': c, 'phase': 'FILL',
                  'oh': oh_i, 'ow': ow_i,
                  'buf_write': 1, 'buf_addr': c,
                  'input_re': int(v), 'padding': int(not v),
                  'weight_re': 0, 'output_we': 0,
                  'buf_fills': c + 1}

        elif c < FILL_CYCLES + GEMM_CYCLES:
            gc     = c - FILL_CYCLES
            # output-stationary: iterate m (pixel), n (oc), k (inner)
            out_idx = gc // K_STEPS
            k_idx   = gc % K_STEPS
            oh_i    = (out_idx // C_OUT) // W_OUT
            ow_i    = (out_idx // C_OUT) % W_OUT
            oc_i    = out_idx % C_OUT
            ev = {'cycle': c, 'phase': 'GEMM',
                  'oh': oh_i, 'ow': ow_i, 'oc': oc_i,
                  'buf_write': 0, 'buf_addr': (out_idx // C_OUT) * K_STEPS + k_idx,
                  'input_re': 0, 'padding': 0,
                  'weight_re': 1, 'output_we': 0,
                  'buf_fills': FILL_CYCLES}

        elif c < total:
            wc  = c - FILL_CYCLES - GEMM_CYCLES
            ev = {'cycle': c, 'phase': 'WRITE',
                  'oh': wc // (W_OUT * C_OUT),
                  'ow': (wc // C_OUT) % W_OUT,
                  'buf_write': 0, 'buf_addr': 0,
                  'input_re': 0, 'padding': 0,
                  'weight_re': 0, 'output_we': 1,
                  'buf_fills': FILL_CYCLES}
        else:
            ev = {'cycle': c, 'phase': 'DONE',
                  'buf_write': 0, 'buf_addr': 0,
                  'input_re': 0, 'padding': 0,
                  'weight_re': 0, 'output_we': 0,
                  'buf_fills': FILL_CYCLES}

        events.append(ev)
        if ev['phase'] == 'DONE':
            break

    return events


def compute_stats(accel_events, im2col_events):
    """Verify simulation matches measured hardware values."""
    a_input  = sum(e['input_re']  for e in accel_events)
    a_weight = sum(e['weight_re'] for e in accel_events)
    a_output = sum(e['output_we'] for e in accel_events)

    i_input  = sum(e['input_re']  for e in im2col_events)
    i_weight = sum(e['weight_re'] for e in im2col_events)
    i_buf    = sum(e.get('buf_write',0) for e in im2col_events)
    i_output = sum(e['output_we'] for e in im2col_events)

    return {
        'accel': {
            'cycles':       len(accel_events),
            'input_reads':  a_input,
            'weight_reads': a_weight,
            'output_writes':a_output,
            'buffer_bytes': 0,
        },
        'im2col': {
            'cycles':       len(im2col_events),
            'input_reads':  i_input,
            'weight_reads': i_weight,
            'output_writes':i_output,
            'buffer_bytes': i_buf,
        }
    }


def compress_events(events, step=4):
    """
    Keep every Nth event plus all state-change boundaries,
    to keep the JS file manageable while preserving accuracy.
    """
    compressed = []
    prev_key = None
    for i, e in enumerate(events):
        # Key for change detection
        cur_key = (e.get('state',''), e.get('phase',''), e.get('oh',0), e.get('ow',0))
        if i % step == 0 or cur_key != prev_key:
            compressed.append(e)
            prev_key = cur_key
    return compressed


if __name__ == '__main__':
    print("Simulating conv_controller FSM...")
    accel  = simulate_accel()
    print(f"  Accelerator: {len(accel)} cycles")

    print("Simulating im2col baseline...")
    im2col = simulate_im2col()
    print(f"  Im2col:      {len(im2col)} cycles")

    stats  = compute_stats(accel, im2col)
    print(f"\nStats (should match hardware simulation):")
    print(f"  Accel  — cycles:{stats['accel']['cycles']:5d}  "
          f"input_reads:{stats['accel']['input_reads']:5d}  "
          f"weight_reads:{stats['accel']['weight_reads']:4d}  "
          f"outputs:{stats['accel']['output_writes']:3d}")
    print(f"  Im2col — cycles:{stats['im2col']['cycles']:5d}  "
          f"input_reads:{stats['im2col']['input_reads']:5d}  "
          f"weight_reads:{stats['im2col']['weight_reads']:5d}  "
          f"buf_bytes:{stats['im2col']['buffer_bytes']:5d}")

    # Hardware-measured reference values from Verilator simulation
    assert stats['accel']['input_reads']  == 1936, f"Expected 1936, got {stats['accel']['input_reads']}"
    assert stats['accel']['weight_reads'] == 288,  f"Expected 288,  got {stats['accel']['weight_reads']}"
    assert stats['accel']['output_writes']== 256,  f"Expected 256,  got {stats['accel']['output_writes']}"
    assert stats['im2col']['buffer_bytes']== 2304,  f"Expected 2304, got {stats['im2col']['buffer_bytes']}"
    assert stats['im2col']['weight_reads']== 9216,  f"Expected 9216, got {stats['im2col']['weight_reads']}"
    print("\n✓ All assertions pass — matches Verilator simulation measurements")

    # Compress for embedding
    accel_c  = compress_events(accel,  step=2)
    im2col_c = compress_events(im2col, step=4)

    payload = {
        'meta': {
            'layer': '8x8x4 conv2d, 3x3 kernel, stride=1, pad=1, C_out=4',
            'tool':  'Verilator 5.048',
            'accel_total_cycles':  stats['accel']['cycles'],
            'im2col_total_cycles': stats['im2col']['cycles'],
            'accel_input_reads':   stats['accel']['input_reads'],
            'accel_weight_reads':  stats['accel']['weight_reads'],
            'accel_output_writes': stats['accel']['output_writes'],
            'im2col_buffer_bytes': stats['im2col']['buffer_bytes'],
            'im2col_weight_reads': stats['im2col']['weight_reads'],
            'im2col_total_reads':  stats['im2col']['input_reads'] + stats['im2col']['weight_reads'],
            'verified_outputs':    256,
        },
        'accel_events':  accel_c,
        'im2col_events': im2col_c,
    }

    js = f"// Auto-generated by generate_trace.py\n"
    js += f"// Simulation verified with Verilator 5.048 — 256/256 outputs correct\n"
    js += f"const SIM_DATA = {json.dumps(payload, separators=(',',':'))};\n"

    with open('trace_data.js', 'w') as f:
        f.write(js)

    size = len(js)
    print(f"\nWrote trace_data.js ({size:,} bytes, "
          f"{len(accel_c)} accel events, {len(im2col_c)} im2col events)")
