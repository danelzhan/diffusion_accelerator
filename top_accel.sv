// =============================================================================
// top_accel.sv  —  Unified Matmul / Conv2d Accelerator Top Level
// =============================================================================
//
// Integrates both compute controllers and the shared 4×4 systolic array behind
// a single start/done interface.  A one-bit mode signal selects the operation:
//
//   mode = 0  →  4×4 INT8 matrix multiplication  (matmul_controller)
//   mode = 1  →  3×3 INT8 conv2d                 (conv_controller)
//
// SRAM allocation
// ---------------
// Three physical SRAM banks are shared across both modes:
//
//   Bank A  (INT8,  DEPTH=256, ALEN=8)
//     mode 0: holds the 4×4 A matrix        (addrs 0..15)
//     mode 1: holds input activations, CHW   (addrs 0..H*W*C-1)
//
//   Bank B  (INT8,  DEPTH=512, ALEN=9)
//     mode 0: holds the 4×4 B matrix        (addrs 0..15)
//     mode 1: holds filter weights, oc×72+k  (addrs 0..C_out×72-1)
//
//   Bank C  (INT32, DEPTH=256, ALEN=8)
//     mode 0: receives the 4×4 result C     (addrs 0..15)
//     mode 1: receives output activations, HWC (addrs 0..H_out×W_out×C_out-1)
//
// Each bank has independent read and write ports (from sram_model).  The
// testbench pre-loads Banks A and B through the external write ports; Bank C
// results are read back through the external read port.
//
// Datapath muxing
// ---------------
// Only the controller matching the current mode receives a start pulse;
// the inactive controller stays in IDLE.  Its SRAM-enable and systolic-array
// drive signals are therefore all deasserted, so no bus conflict occurs.
// The mode mux on each SRAM read/write path is purely combinatorial.
//
// =============================================================================

module top_accel (
    input  logic        clock,
    input  logic        reset,

    // ── Operation select and control ─────────────────────────────────────────
    input  logic        mode,      // 0 = matmul, 1 = conv2d
    input  logic        start,
    output logic        done,

    // ── Conv2d layer configuration (ignored when mode = 0) ───────────────────
    input  logic [7:0]  H_in,
    input  logic [7:0]  W_in,
    input  logic [7:0]  H_out,
    input  logic [7:0]  W_out,
    input  logic [7:0]  C_in,
    input  logic [7:0]  C_out,
    input  logic [3:0]  stride,
    input  logic [3:0]  padding,

    // ── Bank A external write (testbench pre-loads activations / A matrix) ───
    input  logic        ext_a_we,
    input  logic [7:0]  ext_a_waddr,
    input  logic [7:0]  ext_a_wdata,

    // ── Bank B external write (testbench pre-loads weights / B matrix) ───────
    input  logic        ext_b_we,
    input  logic [8:0]  ext_b_waddr,
    input  logic [7:0]  ext_b_wdata,

    // ── Bank C external read (testbench reads results) ────────────────────────
    input  logic        ext_c_re,
    input  logic [7:0]  ext_c_raddr,
    output logic signed [31:0] ext_c_rdata
);

    // =========================================================================
    // Systolic array wires
    // =========================================================================
    logic               sa_clear, sa_valid_in, sa_valid_out;
    logic signed [7:0]  sa_a_in  [4];
    logic signed [7:0]  sa_b_in  [4];
    logic signed [31:0] sa_c_out [4][4];

    // =========================================================================
    // matmul_controller signals
    // =========================================================================
    logic        mm_done, mm_start;
    logic        mm_a_re;
    logic [3:0]  mm_a_raddr;
    logic signed [7:0] mm_a_rdata;
    logic        mm_b_re;
    logic [3:0]  mm_b_raddr;
    logic signed [7:0] mm_b_rdata;
    logic        mm_c_we;
    logic [3:0]  mm_c_waddr;
    logic signed [31:0] mm_c_wdata;
    logic        mm_sa_clear, mm_sa_valid_in;
    logic signed [7:0] mm_sa_a_in [4];
    logic signed [7:0] mm_sa_b_in [4];

    assign mm_start = start && !mode;

    // =========================================================================
    // conv_controller signals
    // =========================================================================
    logic        cv_done, cv_start;
    logic        cv_in_re;
    logic [15:0] cv_in_raddr;
    logic signed [7:0] cv_in_rdata;
    logic        cv_w_re;
    logic [15:0] cv_w_raddr;
    logic signed [7:0] cv_w_rdata;
    logic        cv_out_we;
    logic [15:0] cv_out_waddr;
    logic signed [31:0] cv_out_wdata;
    logic        cv_sa_clear, cv_sa_valid_in;
    logic signed [7:0] cv_sa_a_in [4];
    logic signed [7:0] cv_sa_b_in [4];

    assign cv_start = start && mode;

    // =========================================================================
    // Done: only one controller is active at a time
    // =========================================================================
    assign done = mm_done | cv_done;

    // =========================================================================
    // SRAM internal wires
    // =========================================================================
    // Bank A read side (muxed between controllers)
    logic        sram_a_re;
    logic [7:0]  sram_a_raddr;
    logic signed [7:0] sram_a_rdata;

    // Bank B read side
    logic        sram_b_re;
    logic [8:0]  sram_b_raddr;
    logic signed [7:0] sram_b_rdata;

    // Bank C write side (muxed between controllers)
    logic        sram_c_we;
    logic [7:0]  sram_c_waddr;
    logic signed [31:0] sram_c_wdata;

    // =========================================================================
    // Datapath mux: route controller signals to SRAMs and systolic array
    // =========================================================================
    always_comb begin
        if (mode) begin
            // ── Conv mode ────────────────────────────────────────────────────
            // Bank A (input activations)
            sram_a_re    = cv_in_re;
            sram_a_raddr = cv_in_raddr[7:0];   // conv addrs fit in 8 bits
            // Bank B (weights)
            sram_b_re    = cv_w_re;
            sram_b_raddr = cv_w_raddr[8:0];    // conv weight addrs fit in 9 bits
            // Bank C (output)
            sram_c_we    = cv_out_we;
            sram_c_waddr = cv_out_waddr[7:0];
            sram_c_wdata = cv_out_wdata;
            // Systolic array
            sa_clear    = cv_sa_clear;
            sa_valid_in = cv_sa_valid_in;
            for (int i = 0; i < 4; i++) begin
                sa_a_in[i] = cv_sa_a_in[i];
                sa_b_in[i] = cv_sa_b_in[i];
            end
        end else begin
            // ── Matmul mode ──────────────────────────────────────────────────
            // Bank A (A matrix)
            sram_a_re    = mm_a_re;
            sram_a_raddr = {4'b0, mm_a_raddr};  // 4-bit addr zero-extended
            // Bank B (B matrix)
            sram_b_re    = mm_b_re;
            sram_b_raddr = {5'b0, mm_b_raddr};  // 4-bit addr zero-extended
            // Bank C (result)
            sram_c_we    = mm_c_we;
            sram_c_waddr = {4'b0, mm_c_waddr};
            sram_c_wdata = mm_c_wdata;
            // Systolic array
            sa_clear    = mm_sa_clear;
            sa_valid_in = mm_sa_valid_in;
            for (int i = 0; i < 4; i++) begin
                sa_a_in[i] = mm_sa_a_in[i];
                sa_b_in[i] = mm_sa_b_in[i];
            end
        end
    end

    // Feed rdata back to the correct controller
    assign mm_a_rdata  = sram_a_rdata;
    assign mm_b_rdata  = sram_b_rdata;
    assign cv_in_rdata = sram_a_rdata;
    assign cv_w_rdata  = sram_b_rdata;

    // =========================================================================
    // SRAM instances
    // =========================================================================

    // Bank A: INT8, 256 entries
    sram_model #(.DEPTH(256), .DWIDTH(8)) u_bank_a (
        .clock(clock), .reset(reset),
        .we   (ext_a_we),    .waddr(ext_a_waddr),  .wdata(ext_a_wdata),
        .re   (sram_a_re),   .raddr(sram_a_raddr), .rdata(sram_a_rdata)
    );

    // Bank B: INT8, 512 entries
    sram_model #(.DEPTH(512), .DWIDTH(8)) u_bank_b (
        .clock(clock), .reset(reset),
        .we   (ext_b_we),    .waddr(ext_b_waddr),  .wdata(ext_b_wdata),
        .re   (sram_b_re),   .raddr(sram_b_raddr), .rdata(sram_b_rdata)
    );

    // Bank C: INT32, 256 entries
    sram_model #(.DEPTH(256), .DWIDTH(32)) u_bank_c (
        .clock(clock), .reset(reset),
        .we   (sram_c_we),   .waddr(sram_c_waddr), .wdata(sram_c_wdata),
        .re   (ext_c_re),    .raddr(ext_c_raddr),  .rdata(ext_c_rdata)
    );

    // =========================================================================
    // matmul_controller instance
    // =========================================================================
    matmul_controller u_mm_ctrl (
        .clock     (clock),        .reset  (reset),
        .start     (mm_start),     .done   (mm_done),
        .a_re      (mm_a_re),      .a_raddr(mm_a_raddr),  .a_rdata(mm_a_rdata),
        .b_re      (mm_b_re),      .b_raddr(mm_b_raddr),  .b_rdata(mm_b_rdata),
        .c_we      (mm_c_we),      .c_waddr(mm_c_waddr),  .c_wdata(mm_c_wdata),
        .sa_clear  (mm_sa_clear),  .sa_valid_in(mm_sa_valid_in),
        .sa_a_in   (mm_sa_a_in),   .sa_b_in    (mm_sa_b_in),
        .sa_c_out  (sa_c_out)
    );

    // =========================================================================
    // conv_controller instance
    // =========================================================================
    conv_controller u_cv_ctrl (
        .clock   (clock),   .reset  (reset),
        .start   (cv_start),.done   (cv_done),
        .H_in    (H_in),    .W_in   (W_in),
        .H_out   (H_out),   .W_out  (W_out),
        .C_in    (C_in),    .C_out  (C_out),
        .stride  (stride),  .padding(padding),
        .input_re   (cv_in_re),   .input_raddr (cv_in_raddr),
        .input_rdata(cv_in_rdata),
        .weight_re  (cv_w_re),    .weight_raddr(cv_w_raddr),
        .weight_rdata(cv_w_rdata),
        .output_we  (cv_out_we),  .output_waddr(cv_out_waddr),
        .output_wdata(cv_out_wdata),
        .sa_clear   (cv_sa_clear),  .sa_valid_in(cv_sa_valid_in),
        .sa_a_in    (cv_sa_a_in),   .sa_b_in    (cv_sa_b_in),
        .sa_c_out   (sa_c_out)
    );

    // =========================================================================
    // Systolic array (shared)
    // =========================================================================
    systolic_array_4x4 u_sa (
        .clock    (clock),      .reset    (reset),
        .clear    (sa_clear),   .valid_in (sa_valid_in),
        .a_in     (sa_a_in),    .b_in     (sa_b_in),
        .c_out    (sa_c_out),   .valid_out(sa_valid_out)
    );

endmodule
