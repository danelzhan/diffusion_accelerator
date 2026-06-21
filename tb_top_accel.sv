// =============================================================================
// tb_top_accel.sv  —  Testbench for top_accel (matmul + conv2d modes)
// =============================================================================
//
// Test plan
// ---------
//   Phase A — Matmul (mode=0)
//     Load Bank A and Bank B with mixed-sign 4×4 INT8 matrices.
//     Assert start, wait for done, read Bank C, compare against golden.
//
//   Phase B — Conv2d (mode=1)
//     Load Bank A with 8×8×4 INT8 activations (CHW).
//     Load Bank B with 4×72 INT8 weights (oc×k flat layout).
//     Assert start, wait for done, read Bank C, compare against golden.
//
// Both phases run on the same DUT instance to verify that the mode-switch
// leaves no state residue between operations.
//
// =============================================================================

`timescale 1ns/1ps

module tb_top_accel;

    // =========================================================================
    // Conv2d layer constants
    // =========================================================================
    localparam int H_IN        = 8;
    localparam int W_IN        = 8;
    localparam int H_OUT       = 8;
    localparam int W_OUT       = 8;
    localparam int C_IN        = 4;
    localparam int C_OUT       = 4;
    localparam int STRIDE      = 1;
    localparam int PADDING     = 1;
    localparam int K_STEPS_MAX = 72;
    localparam int K_STEPS     = C_IN * 9;  // 36

    // =========================================================================
    // Clock & reset
    // =========================================================================
    logic clock, reset;
    initial  clock = 1'b0;
    always  #5 clock = ~clock;

    // =========================================================================
    // DUT interface signals
    // =========================================================================
    logic        mode, start, done;

    // Conv config
    logic [7:0]  H_in_sv = 8'(H_IN),  W_in_sv = 8'(W_IN);
    logic [7:0]  H_out_sv= 8'(H_OUT), W_out_sv= 8'(W_OUT);
    logic [7:0]  C_in_sv = 8'(C_IN),  C_out_sv= 8'(C_OUT);
    logic [3:0]  stride_sv = 4'(STRIDE), padding_sv = 4'(PADDING);

    // Bank A write
    logic       ext_a_we;
    logic [7:0] ext_a_waddr;
    logic [7:0] ext_a_wdata;

    // Bank B write
    logic       ext_b_we;
    logic [8:0] ext_b_waddr;
    logic [7:0] ext_b_wdata;

    // Bank C read
    logic        ext_c_re;
    logic [7:0]  ext_c_raddr;
    logic signed [31:0] ext_c_rdata;

    // =========================================================================
    // DUT instantiation
    // =========================================================================
    top_accel u_dut (
        .clock   (clock),   .reset  (reset),
        .mode    (mode),    .start  (start),    .done  (done),
        .H_in    (H_in_sv), .W_in   (W_in_sv),
        .H_out   (H_out_sv),.W_out  (W_out_sv),
        .C_in    (C_in_sv), .C_out  (C_out_sv),
        .stride  (stride_sv),.padding(padding_sv),
        .ext_a_we(ext_a_we),    .ext_a_waddr(ext_a_waddr), .ext_a_wdata(ext_a_wdata),
        .ext_b_we(ext_b_we),    .ext_b_waddr(ext_b_waddr), .ext_b_wdata(ext_b_wdata),
        .ext_c_re(ext_c_re),    .ext_c_raddr(ext_c_raddr), .ext_c_rdata(ext_c_rdata)
    );

    // =========================================================================
    // Waveform
    // =========================================================================
    initial begin
        $dumpfile("dumpfile.fst");
        $dumpvars(0);
    end

    // =========================================================================
    // ── Phase A: Matmul helpers ───────────────────────────────────────────────
    // =========================================================================

    // Reuse test 4 from tb_matmul (mixed-sign)
    logic signed [7:0] A_mat [4][4];
    logic signed [7:0] B_mat [4][4];
    logic signed [31:0] mm_golden [4][4];

    task automatic init_mm_matrices;
        A_mat[0] = '{  8'sd3, -8'sd1,  8'sd5, -8'sd2 };
        A_mat[1] = '{ -8'sd4,  8'sd6, -8'sd3,  8'sd1 };
        A_mat[2] = '{  8'sd2, -8'sd5,  8'sd7, -8'sd4 };
        A_mat[3] = '{ -8'sd1,  8'sd3, -8'sd6,  8'sd8 };
        B_mat[0] = '{  8'sd1, -8'sd2,  8'sd3, -8'sd4 };
        B_mat[1] = '{ -8'sd3,  8'sd5, -8'sd1,  8'sd2 };
        B_mat[2] = '{  8'sd2, -8'sd4,  8'sd6, -8'sd3 };
        B_mat[3] = '{ -8'sd5,  8'sd1, -8'sd2,  8'sd7 };
    endtask

    task automatic compute_mm_golden;
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                mm_golden[i][j] = 32'sd0;
                for (int k = 0; k < 4; k++) begin
                    mm_golden[i][j] += 32'(signed'(A_mat[i][k])) *
                                       32'(signed'(B_mat[k][j]));
                end
            end
        end
    endtask

    task automatic load_mm_banks;
        // Load A matrix into Bank A (row-major: A[i][k] → addr i*4+k)
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                @(posedge clock); #1;
                ext_a_we    = 1'b1;
                ext_a_waddr = 8'(i*4+j);
                ext_a_wdata = A_mat[i][j];
            end
        end
        @(posedge clock); #1; ext_a_we = 1'b0;
        // Load B matrix into Bank B (B[k][j] → addr k*4+j)
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                @(posedge clock); #1;
                ext_b_we    = 1'b1;
                ext_b_waddr = 9'(i*4+j);
                ext_b_wdata = B_mat[i][j];
            end
        end
        @(posedge clock); #1; ext_b_we = 1'b0;
    endtask

    task automatic verify_mm(output int errors);
        logic signed [31:0] actual;
        errors = 0;
        ext_c_re = 1'b1;
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                @(posedge clock); #1;
                ext_c_raddr = 8'(i*4+j);
                @(posedge clock); #1;
                actual = ext_c_rdata;
                if (actual !== mm_golden[i][j]) begin
                    $display("LOG: %0t : ERROR : tb_top_accel : bank_c.mm[%0d][%0d] : expected: %0d actual: %0d",
                             $time, i, j, mm_golden[i][j], int'(actual));
                    errors++;
                end else begin
                    $display("LOG: %0t : INFO  : tb_top_accel : bank_c.mm[%0d][%0d] : expected: %0d actual: %0d",
                             $time, i, j, mm_golden[i][j], int'(actual));
                end
            end
        end
        @(posedge clock); #1; ext_c_re = 1'b0;
    endtask

    // =========================================================================
    // ── Phase B: Conv2d helpers (mirrors tb_conv2d) ───────────────────────────
    // =========================================================================

    task automatic load_conv_banks;
        // Bank A: input activations, CHW, addr=(ic*H+ih)*W+iw, val=(addr%5)-2
        for (int ic = 0; ic < C_IN; ic++) begin
            for (int ih = 0; ih < H_IN; ih++) begin
                for (int iw = 0; iw < W_IN; iw++) begin
                    @(posedge clock); #1;
                    ext_a_we    = 1'b1;
                    ext_a_waddr = 8'((ic * H_IN + ih) * W_IN + iw);
                    ext_a_wdata = 8'(int'(ext_a_waddr) % 5 - 2);
                end
            end
        end
        @(posedge clock); #1; ext_a_we = 1'b0;

        // Bank B: weights, addr=oc*72+k, val=(addr%7)-3; zeros for k>=K_STEPS
        for (int oc = 0; oc < C_OUT; oc++) begin
            for (int k = 0; k < K_STEPS_MAX; k++) begin
                @(posedge clock); #1;
                ext_b_we    = 1'b1;
                ext_b_waddr = 9'(oc * K_STEPS_MAX + k);
                if (k < K_STEPS) begin
                    ext_b_wdata = 8'(int'(ext_b_waddr) % 7 - 3);
                end else begin
                    ext_b_wdata = 8'd0;
                end
            end
        end
        @(posedge clock); #1; ext_b_we = 1'b0;
    endtask

    integer cv_inp  [C_IN][H_IN][W_IN];
    integer cv_wgt  [C_OUT][K_STEPS_MAX];
    integer cv_gold [H_OUT][W_OUT][C_OUT];

    task automatic compute_conv_golden;
        integer addr_i, addr_w, ih_s, iw_s, k;
        for (int ic = 0; ic < C_IN; ic++) begin
            for (int ih = 0; ih < H_IN; ih++) begin
                for (int iw = 0; iw < W_IN; iw++) begin
                    addr_i = (ic * H_IN + ih) * W_IN + iw;
                    cv_inp[ic][ih][iw] = addr_i % 5 - 2;
                end
            end
        end
        for (int oc = 0; oc < C_OUT; oc++) begin
            for (int kk = 0; kk < K_STEPS_MAX; kk++) begin
                addr_w = oc * K_STEPS_MAX + kk;
                cv_wgt[oc][kk] = (kk < K_STEPS) ? (addr_w % 7 - 3) : 0;
            end
        end
        for (int oh = 0; oh < H_OUT; oh++) begin
            for (int ow = 0; ow < W_OUT; ow++) begin
                for (int oc = 0; oc < C_OUT; oc++) begin
                    cv_gold[oh][ow][oc] = 0;
                    for (int ic = 0; ic < C_IN; ic++) begin
                        for (int kr = 0; kr < 3; kr++) begin
                            for (int kc = 0; kc < 3; kc++) begin
                                ih_s = oh * STRIDE + kr - PADDING;
                                iw_s = ow * STRIDE + kc - PADDING;
                                if (ih_s >= 0 && ih_s < H_IN &&
                                    iw_s >= 0 && iw_s < W_IN) begin
                                    k = ic * 9 + kr * 3 + kc;
                                    cv_gold[oh][ow][oc] +=
                                        cv_inp[ic][ih_s][iw_s] * cv_wgt[oc][k];
                                end
                            end
                        end
                    end
                end
            end
        end
    endtask

    task automatic verify_conv(output int errors);
        logic signed [31:0] actual;
        automatic int addr;
        errors = 0;
        ext_c_re = 1'b1;
        for (int oh = 0; oh < H_OUT; oh++) begin
            for (int ow = 0; ow < W_OUT; ow++) begin
                for (int oc = 0; oc < C_OUT; oc++) begin
                    addr = (oh * W_OUT + ow) * C_OUT + oc;
                    @(posedge clock); #1;
                    ext_c_raddr = 8'(addr);
                    @(posedge clock); #1;
                    actual = ext_c_rdata;
                    if (actual !== 32'(cv_gold[oh][ow][oc])) begin
                        $display("LOG: %0t : ERROR : tb_top_accel : bank_c.conv[%0d] : expected: %0d actual: %0d",
                                 $time, addr, cv_gold[oh][ow][oc], int'(actual));
                        errors++;
                    end else begin
                        $display("LOG: %0t : INFO  : tb_top_accel : bank_c.conv[%0d] : expected: %0d actual: %0d",
                                 $time, addr, cv_gold[oh][ow][oc], int'(actual));
                    end
                end
            end
        end
        @(posedge clock); #1; ext_c_re = 1'b0;
    endtask

    // =========================================================================
    // Generic run-and-wait helper
    // =========================================================================
    task automatic run_and_wait(input int timeout_limit);
        int cnt;
        @(posedge clock); #1; start = 1'b1;
        @(posedge clock); #1; start = 1'b0;
        cnt = 0;
        while (!done) begin
            @(posedge clock); #1;
            if (++cnt > timeout_limit) begin
                $display("ERROR"); $fatal(1, "tb_top_accel: timeout");
            end
        end
        repeat (2) @(posedge clock);
    endtask

    // =========================================================================
    // Main test flow
    // =========================================================================
    int total_errors, err_tmp;

    initial begin
        $display("TEST START");
        total_errors = 0;

        // Idle all signals
        reset       = 1'b1;
        mode        = 1'b0;
        start       = 1'b0;
        ext_a_we    = 1'b0; ext_a_waddr = '0; ext_a_wdata = '0;
        ext_b_we    = 1'b0; ext_b_waddr = '0; ext_b_wdata = '0;
        ext_c_re    = 1'b0; ext_c_raddr = '0;

        repeat (4) @(posedge clock);
        #1; reset = 1'b0;
        repeat (2) @(posedge clock);

        // =======================================================
        // Phase A: Matmul  (mode = 0)
        // =======================================================
        $display("\n=== Phase A: Matmul (mode=0) ===");
        mode = 1'b0;

        $display("[A1] Loading Bank A (A matrix) and Bank B (B matrix) ...");
        init_mm_matrices();
        compute_mm_golden();
        load_mm_banks();
        repeat (2) @(posedge clock);

        $display("[A2] Running matmul ...");
        run_and_wait(500);
        $display("[A2] Done at t=%0t", $time);

        $display("[A3] Verifying 16 results ...");
        verify_mm(err_tmp);
        total_errors += err_tmp;
        if (err_tmp == 0) $display("[A3] Matmul PASS — 16/16 outputs correct ✓");
        else              $display("[A3] Matmul FAIL — %0d mismatches", err_tmp);

        // =======================================================
        // Phase B: Conv2d  (mode = 1)
        // =======================================================
        $display("\n=== Phase B: Conv2d (mode=1) ===");
        mode = 1'b1;
        repeat (2) @(posedge clock);

        $display("[B1] Loading Bank A (activations) and Bank B (weights) ...");
        load_conv_banks();
        compute_conv_golden();
        repeat (2) @(posedge clock);

        $display("[B2] Running conv2d ...");
        run_and_wait(15000);
        $display("[B2] Done at t=%0t", $time);

        $display("[B3] Verifying 256 results ...");
        verify_conv(err_tmp);
        total_errors += err_tmp;
        if (err_tmp == 0) $display("[B3] Conv2d PASS — 256/256 outputs correct ✓");
        else              $display("[B3] Conv2d FAIL — %0d mismatches", err_tmp);

        // =======================================================
        // Final verdict
        // =======================================================
        repeat (4) @(posedge clock);
        if (total_errors == 0) begin
            $display("\nTEST PASSED");
            $finish;
        end else begin
            $display("\nERROR");
            $error("tb_top_accel: %0d total mismatch(es)", total_errors);
            $fatal(1, "tb_top_accel FAILED");
        end
    end

    // Watchdog
    initial begin
        #3_000_000;
        $display("ERROR");
        $fatal(1, "tb_top_accel: watchdog timeout");
    end

endmodule
