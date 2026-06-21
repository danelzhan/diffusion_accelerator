// =============================================================================
// conv_addr_gen.sv  —  Convolution Input Address Generator
// =============================================================================
//
// The paper's core hardware idea in its simplest form.
//
// For every (output pixel, input channel, kernel position) query, this module
// computes the flat SRAM address of the corresponding input activation — the
// same address that software im2col would materialise into a large buffer.
// Instead of writing that buffer, we generate its addresses on-the-fly using
// only a handful of multipliers and comparators, with zero intermediate storage.
//
// Operation
// ---------
//   Given:
//     oh, ow   — output spatial coordinates  (the pixel being computed)
//     ic       — input channel index
//     kr, kc   — kernel row and column (0..KH-1, 0..KW-1)
//
//   Compute input spatial coordinates:
//     ih = oh * stride + kr - padding
//     iw = ow * stride + kc - padding
//
//   Bounds check (handles zero-padding):
//     if ih < 0 || ih >= H_in || iw < 0 || iw >= W_in:
//         addr_valid = 0   ← controller injects zero, no SRAM read needed
//     else:
//         sram_addr  = (ic * H_in + ih) * W_in + iw   ← flat row-major address
//         addr_valid = 1
//
// Why this eliminates im2col
// --------------------------
//   A software im2col for an H_out×W_out feature map with C_in channels and
//   a 3×3 kernel produces a (H_out×W_out) × (C_in×9) matrix in memory.
//   For 8×8 input, C_in=4: that's 64 × 36 = 2304 bytes written and then read.
//
//   This module produces the same addresses combinationally from 6 config
//   registers.  The only memory traffic is the original input SRAM reads —
//   no intermediate buffer is ever written or read.
//
// Design notes
// ------------
//   • Purely combinational — no clock or reset required.
//   • query_valid gates both outputs (sram_addr and addr_valid go to 0
//     when query_valid is low, preventing spurious SRAM reads).
//   • Signed 16-bit intermediates are used for ih_s / iw_s so that negative
//     coordinates (padding region) are detected correctly before the unsigned
//     flat-address arithmetic.
//   • Address arithmetic uses 32-bit unsigned intermediates to avoid
//     overflow for large feature maps (up to 256×256×256).
//   • ADDR_W is parameterised; default 16 bits covers up to depth 65535.
//
// Ports
// -----
//   Layer config (held stable for the duration of one conv layer):
//     H_in    [7:0]  : input feature map height  (rows)
//     W_in    [7:0]  : input feature map width   (cols)
//     stride  [3:0]  : convolution stride (1 for MVP)
//     padding [3:0]  : zero-padding amount (1 for 3×3 same-padding)
//
//   Per-cycle query:
//     oh      [7:0]  : output row coordinate
//     ow      [7:0]  : output column coordinate
//     ic      [7:0]  : input channel index
//     kr      [1:0]  : kernel row   (0..2 for 3×3)
//     kc      [1:0]  : kernel column (0..2 for 3×3)
//     query_valid    : qualifies all query inputs; outputs are 0 when low
//
//   Result (combinational, valid 0 ns after inputs settle):
//     sram_addr [ADDR_W-1:0] : flat input SRAM address to read
//     addr_valid             : 1 = valid SRAM read; 0 = zero-pad, skip SRAM
//
// Address layout (must match input SRAM load order in the controller)
// -------------------------------------------------------------------
//   Input SRAM stores activations in channel-major, row-major order:
//     addr = (ic * H_in + ih) * W_in + iw
//
//   Example for H_in=W_in=4, C_in=2:
//     ic=0, ih=0, iw=0 → addr 0
//     ic=0, ih=0, iw=1 → addr 1
//     ...
//     ic=0, ih=3, iw=3 → addr 15
//     ic=1, ih=0, iw=0 → addr 16
//
// =============================================================================

module conv_addr_gen #(
    parameter int ADDR_W = 16   // output address width (bits)
) (
    // ── Layer configuration ───────────────────────────────────────────────────
    input  logic [7:0]  H_in,     // input feature map height
    input  logic [7:0]  W_in,     // input feature map width
    input  logic [3:0]  stride,   // convolution stride
    input  logic [3:0]  padding,  // zero-padding size

    // ── Per-cycle query ───────────────────────────────────────────────────────
    input  logic [7:0]  oh,           // output pixel row
    input  logic [7:0]  ow,           // output pixel column
    input  logic [7:0]  ic,           // input channel
    input  logic [1:0]  kr,           // kernel row   (0..2)
    input  logic [1:0]  kc,           // kernel column (0..2)
    input  logic        query_valid,  // 1 = query inputs are valid

    // ── Combinational result ──────────────────────────────────────────────────
    output logic [ADDR_W-1:0] sram_addr,  // flat input SRAM address
    output logic               addr_valid  // 0 = zero-pad; 1 = read SRAM
);

    // =========================================================================
    // Intermediate signed coordinates
    //
    // ih_s and iw_s are 16-bit signed.
    // Worst-case magnitudes:
    //   positive: oh(255) × stride(15) + kr(2) = 3827  → fits in 12 bits
    //   negative: 0×0 + 0 - padding(15)        = -15   → sign bit needed
    // 16-bit signed covers -32768..+32767, more than sufficient.
    // =========================================================================
    logic signed [15:0] ih_s;   // input row    (may be negative in padding region)
    logic signed [15:0] iw_s;   // input column (may be negative in padding region)

    // =========================================================================
    // Combinational address computation
    // =========================================================================
    always_comb begin
        // ── Step 1: compute input coordinates ────────────────────────────────
        // Extend all unsigned inputs to 16-bit signed before arithmetic so that
        // the result is correctly signed (no unexpected sign-extension from small
        // operands).
        ih_s = ($signed({8'b0, oh})  * $signed({12'b0, stride}))
             + $signed({14'b0, kr})
             - $signed({12'b0, padding});

        iw_s = ($signed({8'b0, ow})  * $signed({12'b0, stride}))
             + $signed({14'b0, kc})
             - $signed({12'b0, padding});

        // ── Step 2: default outputs ───────────────────────────────────────────
        sram_addr = '0;
        addr_valid = 1'b0;

        // ── Step 3: evaluate only when query is valid ─────────────────────────
        if (query_valid) begin
            if (ih_s[15]                              ||   // ih < 0  (sign bit)
                iw_s[15]                              ||   // iw < 0  (sign bit)
                ih_s >= $signed({8'b0, H_in})         ||   // ih >= H_in
                iw_s >= $signed({8'b0, W_in}))  begin      // iw >= W_in

                // Out of bounds: zero-padding region — caller injects 0
                addr_valid = 1'b0;
                sram_addr  = '0;

            end else begin
                // In bounds: flat channel-major row-major address
                // addr = (ic * H_in + ih) * W_in + iw
                // All arithmetic in 32 bits to prevent overflow for large maps.
                addr_valid = 1'b1;
                sram_addr  = ADDR_W'(
                    (32'({24'b0, ic})       * 32'({24'b0, H_in}) +
                     32'({24'b0, ih_s[7:0]}))
                    * 32'({24'b0, W_in})
                    +  32'({24'b0, iw_s[7:0]})
                );
            end
        end
    end

endmodule
