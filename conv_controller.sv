// =============================================================================
// conv_controller.sv  —  3×3 Convolution Controller (address-centric dataflow)
// =============================================================================
//
// Implements the paper's core idea: drives a 3×3 conv through the 4×4
// systolic array WITHOUT materialising an im2col buffer.  For every
// output pixel, conv_addr_gen computes input SRAM addresses on-the-fly;
// out-of-bounds positions are zero-padded without any SRAM read.
//
// Architecture
// ------------
//   • Processes 4 output channels in parallel (one per systolic array column).
//   • Processes one output pixel at a time (only row 0 of the array is active).
//   • Weight SRAM is pre-loaded into local registers (w_reg) at the start of
//     each output-channel tile, eliminating repeated SRAM port contention
//     during the compute phase.
//
// Weight SRAM layout (fixed stride, padding-friendly)
// ---------------------------------------------------
//   Addr = oc * K_STRIDE + k   where k = ic*9 + kr*3 + kc,
//   K_STRIDE = K_STEPS_MAX = 72  (C_IN_MAX × 9 = 8 × 9).
//   Steps k = C_in*9..71 must be zeros in the SRAM (testbench handles this).
//
// Output SRAM layout (HWC, channel-last)
// ---------------------------------------
//   Addr = (oh * W_out + ow) * C_out + oc
//
// FSM
// ---
//   IDLE → PRELOAD → CLEAR → FEED → DRAIN → WRITE → ... → DONE → IDLE
//
//   PRELOAD : Load 4 × K_STEPS_MAX = 288 weights from weight SRAM into w_reg.
//             Duration: K_STEPS_MAX × 4 + 1 = 289 cycles (accounts for
//             the 1-cycle SRAM read latency on the final entry).
//   CLEAR   : 1 cycle.  Assert sa_clear; present SRAM address for step 0
//             of the current pixel so that data is ready at FEED cycle 0.
//   FEED    : K_steps = C_in × 9 cycles.  valid_in=1 each cycle.
//             a_in[0] = rdata from input SRAM (or 0 for padding pixels).
//             b_in[j] = w_reg[j][step_cnt]  (combinatorial register read).
//             During each FEED cycle the address for step+1 is presented to
//             the input SRAM (1-cycle read pipeline).
//   DRAIN   : 6 cycles, valid_in=0.  Waits for the systolic array pipeline.
//   WRITE   : 4 cycles.  Writes c_out[0][0..3] to the output SRAM.
//   Outer loop repeats CLEAR→FEED→DRAIN→WRITE for every (oh, ow) pixel.
//   When all pixels of the current oc_tile are done, PRELOAD for the next.
//
// Constraints / limitations (MVP)
// --------------------------------
//   • C_out must be a multiple of 4.
//   • C_in ≤ 8  (K_steps ≤ K_STEPS_MAX = 72).
//   • Kernel size fixed at 3×3.
//   • Only row 0 of the systolic array is active; rows 1–3 accumulate zeros
//     (they are cleared at the start of each pixel, so c_out[1..3][j] = 0).
//
// =============================================================================

module conv_controller (
    input  logic        clock,
    input  logic        reset,
    input  logic        start,
    output logic        done,

    // ── Layer configuration (stable for the duration of one conv layer) ──────
    input  logic [7:0]  H_in,
    input  logic [7:0]  W_in,
    input  logic [7:0]  H_out,
    input  logic [7:0]  W_out,
    input  logic [7:0]  C_in,
    input  logic [7:0]  C_out,    // must be multiple of 4
    input  logic [3:0]  stride,
    input  logic [3:0]  padding,

    // ── Input activation SRAM (INT8) ─────────────────────────────────────────
    output logic         input_re,
    output logic [15:0]  input_raddr,
    input  logic signed [7:0]  input_rdata,

    // ── Weight SRAM (INT8, layout: W[oc][k] at oc*72+k) ─────────────────────
    output logic         weight_re,
    output logic [15:0]  weight_raddr,
    input  logic signed [7:0]  weight_rdata,

    // ── Output SRAM (INT32, HWC: addr = (oh*W_out+ow)*C_out+oc) ─────────────
    output logic         output_we,
    output logic [15:0]  output_waddr,
    output logic signed [31:0] output_wdata,

    // ── Systolic array ────────────────────────────────────────────────────────
    output logic         sa_clear,
    output logic         sa_valid_in,
    output logic signed [7:0]  sa_a_in [4],
    output logic signed [7:0]  sa_b_in [4],
    input  logic signed [31:0] sa_c_out [4][4]
);

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam int K_STEPS_MAX = 72;   // C_IN_MAX(8) × 9
    localparam int K_STRIDE    = 72;   // weight SRAM stride per output channel

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE    = 3'd0,
        PRELOAD = 3'd1,
        CLEAR   = 3'd2,
        FEED    = 3'd3,
        DRAIN   = 3'd4,
        WRITE   = 3'd5,
        DONE    = 3'd6
    } state_t;

    state_t state, next_state;

    always_ff @(posedge clock) begin
        if (reset) state <= IDLE;
        else       state <= next_state;
    end

    // =========================================================================
    // Derived config (combinatorial — stable while layer params are held)
    // =========================================================================
    logic [7:0] K_steps;
    assign K_steps = C_in * 8'd9;   // number of feed steps per output pixel

    // =========================================================================
    // Weight preload counters
    //   pre_j  : relative output-channel index (0..3) being addressed
    //   pre_k  : weight step index (0..K_STEPS_MAX-1) being addressed
    //   *_d    : 1-cycle delayed copies (for SRAM-latency-aware capture)
    //   pre_valid_d : high when delayed indices are valid for capture
    // =========================================================================
    logic [1:0] pre_j,  pre_j_d;
    logic [6:0] pre_k,  pre_k_d;
    logic       pre_valid_d;

    // Local weight register file  [relative OC][step]
    logic signed [7:0] w_reg [4][K_STEPS_MAX];

    // =========================================================================
    // Outer-loop position registers
    // =========================================================================
    logic [7:0] oh_cnt;     // current output row
    logic [7:0] ow_cnt;     // current output column
    logic [7:0] oc_base;    // current output-channel tile base (0, 4, 8, …)

    // =========================================================================
    // Per-pixel feed counters
    // =========================================================================
    logic [2:0] ic_cnt;     // input channel  (0..C_in-1)
    logic [1:0] kr_cnt;     // kernel row     (0..2)
    logic [1:0] kc_cnt;     // kernel column  (0..2)
    logic [7:0] step_cnt;   // linear step index (0..K_steps-1)

    logic [2:0] drain_cnt;  // drain cycles (0..5)
    logic [1:0] write_cnt;  // write cycles (0..3)

    // 1-cycle pipeline: registered copy of ag_addr_valid
    logic addr_valid_pipe;

    // =========================================================================
    // Internal conv_addr_gen instance
    // =========================================================================
    logic [7:0]  ag_ic;
    logic [1:0]  ag_kr, ag_kc;
    logic        ag_query_valid;
    logic [15:0] ag_sram_addr;
    logic        ag_addr_valid;

    conv_addr_gen #(.ADDR_W(16)) u_addr_gen (
        .H_in        (H_in),
        .W_in        (W_in),
        .stride      (stride),
        .padding     (padding),
        .oh          (oh_cnt),
        .ow          (ow_cnt),
        .ic          (ag_ic),
        .kr          (ag_kr),
        .kc          (ag_kc),
        .query_valid (ag_query_valid),
        .sram_addr   (ag_sram_addr),
        .addr_valid  (ag_addr_valid)
    );

    // =========================================================================
    // Combinational: next (ic, kr, kc) — one step ahead of current counters.
    // Used to pre-fetch the SRAM address for step step_cnt+1 while feeding
    // the data for step step_cnt.
    // =========================================================================
    logic [2:0] next_ic;
    logic [1:0] next_kr, next_kc;

    always_comb begin
        next_kc = (kc_cnt == 2'd2) ? 2'd0 : kc_cnt + 2'd1;
        if (kc_cnt == 2'd2) begin
            next_kr = (kr_cnt == 2'd2) ? 2'd0 : kr_cnt + 2'd1;
        end else begin
            next_kr = kr_cnt;
        end
        if (kc_cnt == 2'd2 && kr_cnt == 2'd2) begin
            next_ic = ic_cnt + 3'd1;
        end else begin
            next_ic = ic_cnt;
        end
    end

    // =========================================================================
    // Addr-gen input mux
    //   CLEAR : address for step 0 of the current pixel (ic=kr=kc=0)
    //   FEED  : address for step step_cnt+1 (one cycle ahead via next_*)
    //           Gated off for the last step (no step K to pre-fetch)
    // =========================================================================
    always_comb begin
        ag_ic          = 8'd0;
        ag_kr          = 2'd0;
        ag_kc          = 2'd0;
        ag_query_valid = 1'b0;
        if (state == CLEAR) begin
            ag_ic          = 8'd0;
            ag_kr          = 2'd0;
            ag_kc          = 2'd0;
            ag_query_valid = 1'b1;
        end else if (state == FEED) begin
            ag_ic          = {5'b0, next_ic};
            ag_kr          = next_kr;
            ag_kc          = next_kc;
            ag_query_valid = (step_cnt < K_steps - 8'd1);
        end
    end

    // Input activation SRAM read (gated by conv_addr_gen's addr_valid)
    assign input_re    = ag_addr_valid;
    assign input_raddr = ag_sram_addr;

    // Pipeline register for addr_valid (accounts for 1-cycle SRAM latency)
    always_ff @(posedge clock) begin
        if (reset) addr_valid_pipe <= 1'b0;
        else       addr_valid_pipe <= ag_addr_valid;
    end

    // =========================================================================
    // Weight SRAM read (PRELOAD state only)
    // =========================================================================
    logic [15:0] preload_oc_idx;
    assign preload_oc_idx = {8'b0, oc_base} + {14'b0, pre_j};
    assign weight_re      = (state == PRELOAD);
    assign weight_raddr   = preload_oc_idx * 16'd72 + {9'b0, pre_k};

    // =========================================================================
    // Weight capture into w_reg (1-cycle SRAM latency)
    // =========================================================================
    always_ff @(posedge clock) begin
        pre_j_d     <= pre_j;
        pre_k_d     <= pre_k;
        pre_valid_d <= (state == PRELOAD);
    end

    always_ff @(posedge clock) begin
        if (pre_valid_d) begin
            w_reg[pre_j_d][pre_k_d] <= weight_rdata;
        end
    end

    // =========================================================================
    // Preload counter  (pre_j × pre_k nested, reset when not in PRELOAD)
    // =========================================================================
    always_ff @(posedge clock) begin
        if (reset || state != PRELOAD) begin
            pre_j <= 2'd0;
            pre_k <= 7'd0;
        end else begin
            if (pre_k == 7'(K_STEPS_MAX - 1)) begin
                pre_k <= 7'd0;
                pre_j <= pre_j + 2'd1;   // natural 2-bit wrap (0→1→2→3→0)
            end else begin
                pre_k <= pre_k + 7'd1;
            end
        end
    end

    // =========================================================================
    // Outer loop counters (oh, ow, oc_base)  — advance at end of WRITE
    // =========================================================================
    always_ff @(posedge clock) begin
        if (reset || state == IDLE) begin
            oh_cnt  <= 8'd0;
            ow_cnt  <= 8'd0;
            oc_base <= 8'd0;
        end else if (state == WRITE && write_cnt == 2'd3) begin
            if (ow_cnt < W_out - 8'd1) begin
                ow_cnt <= ow_cnt + 8'd1;
            end else begin
                ow_cnt <= 8'd0;
                if (oh_cnt < H_out - 8'd1) begin
                    oh_cnt <= oh_cnt + 8'd1;
                end else begin
                    oh_cnt  <= 8'd0;
                    oc_base <= oc_base + 8'd4;
                end
            end
        end
    end

    // =========================================================================
    // Per-pixel feed counters  (reset in any non-FEED state)
    // =========================================================================
    always_ff @(posedge clock) begin
        if (reset || state != FEED) begin
            step_cnt <= 8'd0;
            ic_cnt   <= 3'd0;
            kr_cnt   <= 2'd0;
            kc_cnt   <= 2'd0;
        end else begin
            step_cnt <= step_cnt + 8'd1;
            if (kc_cnt == 2'd2) begin
                kc_cnt <= 2'd0;
                if (kr_cnt == 2'd2) begin
                    kr_cnt <= 2'd0;
                    ic_cnt <= ic_cnt + 3'd1;
                end else begin
                    kr_cnt <= kr_cnt + 2'd1;
                end
            end else begin
                kc_cnt <= kc_cnt + 2'd1;
            end
        end
    end

    always_ff @(posedge clock) begin
        if (reset || state != DRAIN) drain_cnt <= 3'd0;
        else                         drain_cnt <= drain_cnt + 3'd1;
    end

    always_ff @(posedge clock) begin
        if (reset || state != WRITE) write_cnt <= 2'd0;
        else                         write_cnt <= write_cnt + 2'd1;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start) next_state = PRELOAD;
            end

            PRELOAD: begin
                // Transition when the last address has been presented.
                // The last weight (w_reg[3][K_STEPS_MAX-1]) will be captured
                // during the first CLEAR cycle via the delayed pre_valid_d path.
                if (pre_j == 2'd3 && pre_k == 7'(K_STEPS_MAX - 1)) begin
                    next_state = CLEAR;
                end
            end

            CLEAR: begin
                next_state = FEED;
            end

            FEED: begin
                if (step_cnt == K_steps - 8'd1) next_state = DRAIN;
            end

            DRAIN: begin
                if (drain_cnt == 3'd5) next_state = WRITE;
            end

            WRITE: begin
                if (write_cnt == 2'd3) begin
                    // More pixels remain in this oc_tile?
                    if (ow_cnt < W_out - 8'd1 || oh_cnt < H_out - 8'd1) begin
                        next_state = CLEAR;
                    // More oc_tiles remain?
                    end else if (oc_base + 8'd4 < C_out) begin
                        next_state = PRELOAD;
                    end else begin
                        next_state = DONE;
                    end
                end
            end

            DONE: begin
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Systolic array — activation inputs
    //   Row 0 : live pixel activation (or 0 for padding)
    //   Rows 1–3 : zero (unused; cleared each pixel by sa_clear)
    // =========================================================================
    assign sa_clear    = (state == CLEAR);
    assign sa_valid_in = (state == FEED);

    always_comb begin
        for (int r = 0; r < 4; r++) begin
            sa_a_in[r] = 8'sd0;
        end
        if (state == FEED) begin
            sa_a_in[0] = addr_valid_pipe ? input_rdata : 8'sd0;
        end
    end

    // =========================================================================
    // Systolic array — weight inputs (from preloaded register file)
    // =========================================================================
    always_comb begin
        for (int c = 0; c < 4; c++) begin
            sa_b_in[c] = 8'sd0;
        end
        if (state == FEED) begin
            for (int c = 0; c < 4; c++) begin
                sa_b_in[c] = w_reg[c][step_cnt[6:0]];
            end
        end
    end

    // =========================================================================
    // Output SRAM write  (row 0 of systolic array, all 4 columns)
    //   addr = (oh_cnt × W_out + ow_cnt) × C_out + oc_base + write_cnt
    // =========================================================================
    always_comb begin
        output_we    = (state == WRITE);
        output_waddr = 16'(
            (32'({24'b0, oh_cnt}) * 32'({24'b0, W_out}) + 32'({24'b0, ow_cnt}))
            * 32'({24'b0, C_out})
            + 32'({24'b0, oc_base})
            + 32'({30'b0, write_cnt})
        );
        output_wdata = sa_c_out[0][write_cnt];
    end

    // =========================================================================
    // Done
    // =========================================================================
    assign done = (state == DONE);

endmodule
