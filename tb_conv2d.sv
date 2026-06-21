// =============================================================================
// tb_conv2d.sv  —  End-to-end Testbench for conv_controller
// =============================================================================
//
// Verification strategy
// ---------------------
// Instantiates the full conv pipeline:
//   sram_model (input, INT8, CHW)
//   sram_model (weight, INT8, flat oc×k)
//   conv_controller  (address-centric dataflow)
//   systolic_array_4x4
//   sram_model (output, INT32, HWC)
//
// Test layer: 8×8×4 input, 3×3 kernel, C_out=4, stride=1, same-padding.
//
// Steps
// -----
//   1. Pre-load input SRAM (CHW layout) with a deterministic pattern.
//   2. Pre-load weight SRAM (oc×k flat layout) with a deterministic pattern.
//   3. Assert start; wait for done.
//   4. Compute golden reference in software (integer arithmetic).
//   5. Read output SRAM and compare with golden (256 INT32 values).
//   6. Print memory-traffic comparison vs im2col.
//
// SRAM layouts
// ------------
//   Input  : CHW — addr = (ic * H_IN + ih) * W_IN + iw          (DEPTH 256)
//   Weight : flat — addr = oc * K_STRIDE(72) + ic*9 + kr*3 + kc (DEPTH 512)
//   Output : HWC — addr = (oh * W_OUT + ow) * C_OUT + oc        (DEPTH 256)
//
// Data patterns
// -------------
//   Input  value at addr a: (a % 5) - 2   → range [-2, 2]
//   Weight value at addr a: (a % 7) - 3   → range [-3, 3]
//   Weight steps k = C_IN*9..71 written as 0 (controller ignores them).
//
// =============================================================================

`timescale 1ns/1ps

module tb_conv2d;

    // =========================================================================
    // Layer constants
    // =========================================================================
    localparam int H_IN        = 8;
    localparam int W_IN        = 8;
    localparam int H_OUT       = 8;   // same-padding, stride=1
    localparam int W_OUT       = 8;
    localparam int C_IN        = 4;
    localparam int C_OUT       = 4;   // must be multiple of 4
    localparam int STRIDE      = 1;
    localparam int PADDING     = 1;
    localparam int K_STEPS_MAX = 72;  // C_IN_MAX(8) × 9
    localparam int K_STEPS     = C_IN * 9;   // 36

    // =========================================================================
    // Clock & reset
    // =========================================================================
    logic clock, reset;
    initial  clock = 1'b0;
    always  #5 clock = ~clock;  // 100 MHz

    // =========================================================================
    // DUT control signals
    // =========================================================================
    logic start, done;

    // ── Input activation SRAM ────────────────────────────────────────────────
    logic       tb_in_we;
    logic [7:0] tb_in_waddr;
    logic [7:0] tb_in_wdata;
    logic        ctrl_in_re;
    logic [15:0] ctrl_in_raddr;
    logic signed [7:0] ctrl_in_rdata;

    // ── Weight SRAM ───────────────────────────────────────────────────────────
    logic       tb_w_we;
    logic [8:0] tb_w_waddr;
    logic [7:0] tb_w_wdata;
    logic        ctrl_w_re;
    logic [15:0] ctrl_w_raddr;
    logic signed [7:0] ctrl_w_rdata;

    // ── Output SRAM ───────────────────────────────────────────────────────────
    logic        ctrl_out_we;
    logic [15:0] ctrl_out_waddr;
    logic signed [31:0] ctrl_out_wdata;
    logic       tb_out_re;
    logic [7:0] tb_out_raddr;
    logic signed [31:0] tb_out_rdata;

    // ── Systolic array wires ──────────────────────────────────────────────────
    logic        sa_clear, sa_valid_in, sa_valid_out;
    logic signed [7:0]  sa_a_in  [4];
    logic signed [7:0]  sa_b_in  [4];
    logic signed [31:0] sa_c_out [4][4];

    // =========================================================================
    // SRAM instances
    // =========================================================================
    // Input: DEPTH=256, DWIDTH=8  →  ALEN=8
    sram_model #(.DEPTH(256), .DWIDTH(8)) u_in_sram (
        .clock(clock), .reset(reset),
        .we   (tb_in_we),              .waddr(tb_in_waddr),
        .wdata(tb_in_wdata),
        .re   (ctrl_in_re),            .raddr(ctrl_in_raddr[7:0]),
        .rdata(ctrl_in_rdata)
    );

    // Weight: DEPTH=512, DWIDTH=8  →  ALEN=9  (holds up to 288 entries)
    sram_model #(.DEPTH(512), .DWIDTH(8)) u_w_sram (
        .clock(clock), .reset(reset),
        .we   (tb_w_we),               .waddr(tb_w_waddr),
        .wdata(tb_w_wdata),
        .re   (ctrl_w_re),             .raddr(ctrl_w_raddr[8:0]),
        .rdata(ctrl_w_rdata)
    );

    // Output: DEPTH=256, DWIDTH=32  →  ALEN=8
    sram_model #(.DEPTH(256), .DWIDTH(32)) u_out_sram (
        .clock(clock), .reset(reset),
        .we   (ctrl_out_we),           .waddr(ctrl_out_waddr[7:0]),
        .wdata(ctrl_out_wdata),
        .re   (tb_out_re),             .raddr(tb_out_raddr),
        .rdata(tb_out_rdata)
    );

    // =========================================================================
    // conv_controller instance
    // =========================================================================
    conv_controller u_ctrl (
        .clock(clock), .reset(reset), .start(start), .done(done),
        .H_in   (8'(H_IN)),   .W_in   (8'(W_IN)),
        .H_out  (8'(H_OUT)),  .W_out  (8'(W_OUT)),
        .C_in   (8'(C_IN)),   .C_out  (8'(C_OUT)),
        .stride (4'(STRIDE)), .padding(4'(PADDING)),
        .input_re   (ctrl_in_re),    .input_raddr (ctrl_in_raddr),
        .input_rdata(ctrl_in_rdata),
        .weight_re  (ctrl_w_re),     .weight_raddr(ctrl_w_raddr),
        .weight_rdata(ctrl_w_rdata),
        .output_we  (ctrl_out_we),   .output_waddr(ctrl_out_waddr),
        .output_wdata(ctrl_out_wdata),
        .sa_clear   (sa_clear),      .sa_valid_in (sa_valid_in),
        .sa_a_in    (sa_a_in),       .sa_b_in     (sa_b_in),
        .sa_c_out   (sa_c_out)
    );

    // =========================================================================
    // systolic_array_4x4 instance
    // =========================================================================
    systolic_array_4x4 u_sa (
        .clock    (clock),     .reset(reset),
        .clear    (sa_clear),  .valid_in(sa_valid_in),
        .a_in     (sa_a_in),   .b_in    (sa_b_in),
        .c_out    (sa_c_out),  .valid_out(sa_valid_out)
    );

    // =========================================================================
    // Waveform dump
    // =========================================================================
    initial begin
        $dumpfile("dumpfile.fst");
        $dumpvars(0);
    end

    // =========================================================================
    // Memory-traffic counters  (count SRAM accesses during conv operation)
    // =========================================================================
    int input_read_cnt;
    int weight_read_cnt;
    int output_write_cnt;

    always_ff @(posedge clock) begin
        if (reset) begin
            input_read_cnt   <= 0;
            weight_read_cnt  <= 0;
            output_write_cnt <= 0;
        end else begin
            if (ctrl_in_re)  input_read_cnt   <= input_read_cnt  + 1;
            if (ctrl_w_re)   weight_read_cnt  <= weight_read_cnt + 1;
            if (ctrl_out_we) output_write_cnt <= output_write_cnt + 1;
        end
    end

    // =========================================================================
    // SRAM load tasks
    // =========================================================================

    // Load input SRAM: CHW layout, addr = (ic*H+ih)*W+iw, val = (addr%5)-2
    task automatic load_input_sram;
        for (int ic = 0; ic < C_IN; ic++) begin
            for (int ih = 0; ih < H_IN; ih++) begin
                for (int iw = 0; iw < W_IN; iw++) begin
                    @(posedge clock); #1;
                    tb_in_we    = 1'b1;
                    tb_in_waddr = 8'((ic * H_IN + ih) * W_IN + iw);
                    tb_in_wdata = 8'(int'(tb_in_waddr) % 5 - 2);
                end
            end
        end
        @(posedge clock); #1; tb_in_we = 1'b0;
    endtask

    // Load weight SRAM: addr = oc*72 + k, val = (addr%7)-3
    // Steps k >= C_IN*9 are loaded as zero (padding, never used in FEED)
    task automatic load_weight_sram;
        for (int oc = 0; oc < C_OUT; oc++) begin
            for (int k = 0; k < K_STEPS_MAX; k++) begin
                @(posedge clock); #1;
                tb_w_we    = 1'b1;
                tb_w_waddr = 9'(oc * K_STEPS_MAX + k);
                if (k < K_STEPS) begin
                    tb_w_wdata = 8'(int'(tb_w_waddr) % 7 - 3);
                end else begin
                    tb_w_wdata = 8'd0;
                end
            end
        end
        @(posedge clock); #1; tb_w_we = 1'b0;
    endtask

    // =========================================================================
    // Golden reference
    // =========================================================================
    integer inp_arr [C_IN][H_IN][W_IN];
    integer wgt_arr [C_OUT][K_STEPS_MAX];
    integer golden  [H_OUT][W_OUT][C_OUT];

    task automatic compute_golden;
        integer addr_i, addr_w, ih_s, iw_s, k;
        // Mirror the SRAM-fill patterns
        for (int ic = 0; ic < C_IN; ic++) begin
            for (int ih = 0; ih < H_IN; ih++) begin
                for (int iw = 0; iw < W_IN; iw++) begin
                    addr_i = (ic * H_IN + ih) * W_IN + iw;
                    inp_arr[ic][ih][iw] = addr_i % 5 - 2;
                end
            end
        end
        for (int oc = 0; oc < C_OUT; oc++) begin
            for (int kk = 0; kk < K_STEPS_MAX; kk++) begin
                addr_w = oc * K_STEPS_MAX + kk;
                wgt_arr[oc][kk] = (kk < K_STEPS) ? (addr_w % 7 - 3) : 0;
            end
        end
        // Compute convolution  (output: HWC, no bias)
        for (int oh = 0; oh < H_OUT; oh++) begin
            for (int ow = 0; ow < W_OUT; ow++) begin
                for (int oc = 0; oc < C_OUT; oc++) begin
                    golden[oh][ow][oc] = 0;
                    for (int ic = 0; ic < C_IN; ic++) begin
                        for (int kr = 0; kr < 3; kr++) begin
                            for (int kc = 0; kc < 3; kc++) begin
                                ih_s = oh * STRIDE + kr - PADDING;
                                iw_s = ow * STRIDE + kc - PADDING;
                                if (ih_s >= 0 && ih_s < H_IN &&
                                    iw_s >= 0 && iw_s < W_IN) begin
                                    k = ic * 9 + kr * 3 + kc;
                                    golden[oh][ow][oc] +=
                                        inp_arr[ic][ih_s][iw_s] * wgt_arr[oc][k];
                                end
                            end
                        end
                    end
                end
            end
        end
    endtask

    // =========================================================================
    // Output verification (reads output SRAM after done)
    // =========================================================================
    task automatic verify_output(output int errors);
        logic signed [31:0] actual;
        automatic int addr;
        errors = 0;
        tb_out_re = 1'b1;
        for (int oh = 0; oh < H_OUT; oh++) begin
            for (int ow = 0; ow < W_OUT; ow++) begin
                for (int oc = 0; oc < C_OUT; oc++) begin
                    addr = (oh * W_OUT + ow) * C_OUT + oc;
                    @(posedge clock); #1;
                    tb_out_raddr = 8'(addr);
                    @(posedge clock); #1;
                    actual = tb_out_rdata;
                    if (actual !== 32'(golden[oh][ow][oc])) begin
                        $display("LOG: %0t : ERROR : tb_conv2d : u_out_sram.mem[%0d] : expected_value: %0d actual_value: %0d",
                                 $time, addr, golden[oh][ow][oc], int'(actual));
                        errors++;
                    end else begin
                        $display("LOG: %0t : INFO  : tb_conv2d : u_out_sram.mem[%0d] : expected_value: %0d actual_value: %0d",
                                 $time, addr, golden[oh][ow][oc], int'(actual));
                    end
                end
            end
        end
        @(posedge clock); #1; tb_out_re = 1'b0;
    endtask

    // =========================================================================
    // Memory-traffic comparison report
    // =========================================================================
    task automatic print_traffic_report;
        automatic int im2col_buf    = H_OUT * W_OUT * C_IN * 9; // 2304 INT8
        automatic int im2col_w_rds  = C_OUT * K_STEPS * H_OUT * W_OUT; // 9216
        automatic int im2col_total  = im2col_buf + im2col_w_rds;

        $display("");
        $display("╔══════════════════════════════════════════════════╗");
        $display("║           Memory Traffic Comparison              ║");
        $display("╠══════════════════════════════════════════════════╣");
        $display("║  Layer : %0dx%0dx%0d → 3×3 conv → %0dx%0dx%0d   ",
                 H_IN, W_IN, C_IN, H_OUT, W_OUT, C_OUT);
        $display("║  stride=%0d  padding=%0d", STRIDE, PADDING);
        $display("╠══════════════════════════════════════════════════╣");
        $display("║  Address-centric (this hardware)                 ║");
        $display("║  Input SRAM reads  : %5d  (unique valid pixels)  ", input_read_cnt);
        $display("║  Weight SRAM reads : %5d  (preload, 4×72)         ", weight_read_cnt);
        $display("║  Output SRAM writes: %5d                           ", output_write_cnt);
        $display("║  Intermediate buf  :     0 bytes  ← key saving    ║");
        $display("╠══════════════════════════════════════════════════╣");
        $display("║  Im2col equivalent (software baseline)           ║");
        $display("║  Im2col buffer     : %5d INT8 bytes              ", im2col_buf);
        $display("║  Im2col fill reads : %5d  (input → temp buffer)  ", im2col_buf);
        $display("║  GEMM weight reads : %5d  (weights × all pixels) ", im2col_w_rds);
        $display("║  Total SRAM reads  : %5d                          ", im2col_total);
        $display("╠══════════════════════════════════════════════════╣");
        $display("║  Savings                                         ║");
        $display("║  Intermediate buf eliminated : %5d bytes (100%%) ", im2col_buf);
        $display("║  Weight re-reads avoided     : %5d reads          ",
                 im2col_w_rds - weight_read_cnt);
        $display("╚══════════════════════════════════════════════════╝");
        $display("");
    endtask

    // =========================================================================
    // Main test flow
    // =========================================================================
    int total_errors, err_tmp;

    initial begin
        $display("TEST START");
        total_errors = 0;

        // Initialise idle signals
        reset        = 1'b1;
        start        = 1'b0;
        tb_in_we     = 1'b0;  tb_in_waddr  = '0;  tb_in_wdata  = '0;
        tb_w_we      = 1'b0;  tb_w_waddr   = '0;  tb_w_wdata   = '0;
        tb_out_re    = 1'b0;  tb_out_raddr = '0;

        repeat (4) @(posedge clock);
        #1; reset = 1'b0;
        repeat (2) @(posedge clock);

        // ── Phase 1: Pre-load SRAMs ─────────────────────────────────────────
        $display("\n[Phase 1] Loading input SRAM  (%0d × INT8) ...",
                 H_IN * W_IN * C_IN);
        load_input_sram();

        $display("[Phase 1] Loading weight SRAM (%0d × INT8) ...",
                 C_OUT * K_STEPS_MAX);
        load_weight_sram();

        repeat (2) @(posedge clock);

        // ── Phase 2: Run hardware conv ──────────────────────────────────────
        $display("\n[Phase 2] Starting conv2d ...");
        @(posedge clock); #1;
        start = 1'b1;
        @(posedge clock); #1;
        start = 1'b0;

        begin : blk_wait
            int timeout_cnt;
            timeout_cnt = 0;
            while (!done) begin
                @(posedge clock); #1;
                timeout_cnt++;
                if (timeout_cnt > 15000) begin
                    $display("ERROR");
                    $fatal(1, "tb_conv2d: timeout waiting for done (>15000 cycles)");
                end
            end
        end
        $display("[Phase 2] Done at t=%0t  (cycle ~%0t)", $time, $time/10);

        repeat (2) @(posedge clock);

        // ── Phase 3: Compute golden reference ──────────────────────────────
        $display("\n[Phase 3] Computing golden reference ...");
        compute_golden();
        $display("[Phase 3] Golden reference ready.");

        // ── Phase 4: Verify output ──────────────────────────────────────────
        $display("\n[Phase 4] Verifying %0d output values ...",
                 H_OUT * W_OUT * C_OUT);
        verify_output(err_tmp);
        total_errors += err_tmp;
        if (err_tmp == 0) begin
            $display("[Phase 4] All %0d outputs match golden ✓",
                     H_OUT * W_OUT * C_OUT);
        end else begin
            $display("[Phase 4] %0d MISMATCH(ES) detected!", err_tmp);
        end

        // ── Phase 5: Print traffic report ───────────────────────────────────
        print_traffic_report();

        // ── Final verdict ────────────────────────────────────────────────────
        repeat (4) @(posedge clock);
        if (total_errors == 0) begin
            $display("TEST PASSED");
            $finish;
        end else begin
            $display("ERROR");
            $error("tb_conv2d: %0d output mismatch(es)", total_errors);
            $fatal(1, "tb_conv2d FAILED");
        end
    end

    // Watchdog
    initial begin
        #2_000_000;
        $display("ERROR");
        $fatal(1, "tb_conv2d: watchdog timeout at 2 ms");
    end

endmodule
