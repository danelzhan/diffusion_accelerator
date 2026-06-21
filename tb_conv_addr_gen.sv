// =============================================================================
// tb_conv_addr_gen.sv  —  Testbench for conv_addr_gen
// =============================================================================
//
// Verification strategy
// ---------------------
// conv_addr_gen is purely combinational, so each test applies inputs and
// checks outputs after a small propagation delay (#1) — no FSM wait loops.
//
// A SystemVerilog golden reference function (ref_addr) mirrors the address
// formula exactly, allowing exhaustive sweeps with automatic checking.
//
// Test groups
// -----------
//   1. query_valid gating — outputs must be 0 when query_valid=0
//   2. Zero-padding corners — all 8 boundary conditions that produce addr_valid=0
//   3. Address formula spot-checks — hand-computed expected values
//   4. Stride=2 operation — verify scaled coordinate arithmetic
//   5. Exhaustive sweep (H=4, W=4, stride=1, pad=1) — all 4×4×9×2 = 288
//      (oh, ow, kr, kc, ic) combinations checked against ref_addr
//   6. Exhaustive sweep (H=8, W=8, stride=1, pad=1) — all 8×8×9×1 = 576
//      combinations with a larger feature map
//
// Address layout under test (channel-major row-major)
// ---------------------------------------------------
//   addr = (ic * H_in + ih) * W_in + iw
//   where  ih = oh * stride + kr - padding
//          iw = ow * stride + kc - padding
//
// =============================================================================

`timescale 1ns/1ps

module tb_conv_addr_gen;

    // =========================================================================
    // Clock  (purely for waveform structure; DUT is combinational)
    // =========================================================================
    logic clock;
    initial  clock = 1'b0;
    always  #5 clock = ~clock;   // 10 ns period

    // =========================================================================
    // DUT interface signals
    // =========================================================================
    logic [7:0]  H_in, W_in;
    logic [3:0]  stride, padding;
    logic [7:0]  oh, ow, ic;
    logic [1:0]  kr, kc;
    logic        query_valid;
    logic [15:0] sram_addr;
    logic        addr_valid;

    // =========================================================================
    // DUT instantiation
    // =========================================================================
    conv_addr_gen #(.ADDR_W(16)) u_dut (
        .H_in        (H_in),
        .W_in        (W_in),
        .stride      (stride),
        .padding     (padding),
        .oh          (oh),
        .ow          (ow),
        .ic          (ic),
        .kr          (kr),
        .kc          (kc),
        .query_valid (query_valid),
        .sram_addr   (sram_addr),
        .addr_valid  (addr_valid)
    );

    // =========================================================================
    // Waveform dump
    // =========================================================================
    initial begin
        $dumpfile("dumpfile.fst");
        $dumpvars(0);
    end

    // =========================================================================
    // Software golden reference model
    // =========================================================================
    function automatic void ref_addr(
        input  int h_in_i, w_in_i, stride_i, padding_i,
        input  int oh_i, ow_i, ic_i, kr_i, kc_i,
        output int exp_addr, exp_valid
    );
        int ih, iw;
        ih = oh_i * stride_i + kr_i - padding_i;
        iw = ow_i * stride_i + kc_i - padding_i;
        if (ih < 0 || ih >= h_in_i || iw < 0 || iw >= w_in_i) begin
            exp_valid = 0;
            exp_addr  = 0;
        end else begin
            exp_valid = 1;
            exp_addr  = (ic_i * h_in_i + ih) * w_in_i + iw;
        end
    endfunction

    // =========================================================================
    // Checking task
    //   Drives DUT inputs, waits for combinational settling, compares outputs.
    //   Returns 1 on mismatch, 0 on match.
    // =========================================================================
    task automatic check_query(
        input  int  h, w, s, p, o_h, o_w, c, k_r, k_c,
        input  int  exp_addr_i, exp_valid_i,
        input  string label,
        output int  err
    );
        int got_addr, got_valid;
        err = 0;

        // Drive inputs
        H_in    = 8'(h);    W_in   = 8'(w);
        stride  = 4'(s);    padding = 4'(p);
        oh      = 8'(o_h);  ow     = 8'(o_w);
        ic      = 8'(c);
        kr      = 2'(k_r);  kc     = 2'(k_c);
        query_valid = 1'b1;

        #1; // allow combinational logic to settle

        got_addr  = int'(sram_addr);
        got_valid = int'(addr_valid);

        if (got_valid !== exp_valid_i || (exp_valid_i && got_addr !== exp_addr_i)) begin
            $display("LOG: %0t : ERROR : tb_conv_addr_gen : dut.sram_addr/addr_valid : expected_value: valid=%0d addr=%0d actual_value: valid=%0d addr=%0d  [%s]",
                     $time, exp_valid_i, exp_addr_i, got_valid, got_addr, label);
            err = 1;
        end else begin
            $display("LOG: %0t : INFO  : tb_conv_addr_gen : dut.sram_addr/addr_valid : expected_value: valid=%0d addr=%0d actual_value: valid=%0d addr=%0d  [%s]",
                     $time, exp_valid_i, exp_addr_i, got_valid, got_addr, label);
        end
    endtask

    // =========================================================================
    // Main stimulus
    // =========================================================================
    int total_errors, tc_errors;
    int exp_a, exp_v;

    initial begin
        $display("TEST START");
        total_errors = 0;

        // Initialise to safe defaults
        H_in = 8'd4;  W_in = 8'd4;
        stride = 4'd1; padding = 4'd1;
        oh = '0; ow = '0; ic = '0;
        kr = '0; kc = '0;
        query_valid = 1'b0;
        #2;

        // =================================================================
        // Group 1: query_valid gating
        //   Outputs must be 0 when query_valid=0 regardless of inputs.
        // =================================================================
        $display("\n=== Group 1: query_valid gating ===");
        query_valid = 1'b0;
        oh = 8'd2; ow = 8'd2; ic = 8'd0; kr = 2'd1; kc = 2'd1;
        #1;
        tc_errors = 0;
        if (sram_addr !== '0 || addr_valid !== 1'b0) begin
            $display("LOG: %0t : ERROR : tb_conv_addr_gen : dut.query_valid_gate : expected_value: addr=0 valid=0 actual_value: addr=%0d valid=%0d",
                     $time, sram_addr, addr_valid);
            tc_errors++;
        end else begin
            $display("LOG: %0t : INFO  : tb_conv_addr_gen : dut.query_valid_gate : expected_value: addr=0 valid=0 actual_value: addr=%0d valid=%0d",
                     $time, sram_addr, addr_valid);
        end
        total_errors += tc_errors;
        if (tc_errors == 0) $display("[Group1_QueryValidGating] PASS");
        else                $display("[Group1_QueryValidGating] FAIL");

        // =================================================================
        // Group 2: Zero-padding boundary cases
        //   H_in=W_in=4, stride=1, padding=1 (3×3 same-pad)
        //   Test all 8 boundary-condition types that produce addr_valid=0.
        // =================================================================
        $display("\n=== Group 2: Boundary / zero-padding cases (H=W=4, s=1, p=1) ===");
        H_in = 8'd4; W_in = 8'd4; stride = 4'd1; padding = 4'd1;
        tc_errors = 0;

        // Top-left corner: oh=0, ow=0, kr=0, kc=0  → ih=-1, iw=-1  INVALID
        check_query(4,4,1,1, 0,0, 0, 0,0,  0,0, "topleft_corner", exp_v);
        tc_errors += exp_v;

        // Top edge: oh=0, ow=2, kr=0, kc=1  → ih=-1, iw=2  INVALID (ih<0)
        check_query(4,4,1,1, 0,2, 0, 0,1,  0,0, "top_edge", exp_v);
        tc_errors += exp_v;

        // Left edge: oh=2, ow=0, kr=1, kc=0 → ih=2, iw=-1  INVALID (iw<0)
        check_query(4,4,1,1, 2,0, 0, 1,0,  0,0, "left_edge", exp_v);
        tc_errors += exp_v;

        // Bottom edge: oh=3, ow=2, kr=2, kc=1 → ih=4, iw=2  INVALID (ih>=H)
        check_query(4,4,1,1, 3,2, 0, 2,1,  0,0, "bottom_edge", exp_v);
        tc_errors += exp_v;

        // Right edge: oh=2, ow=3, kr=1, kc=2 → ih=2, iw=4  INVALID (iw>=W)
        check_query(4,4,1,1, 2,3, 0, 1,2,  0,0, "right_edge", exp_v);
        tc_errors += exp_v;

        // Bottom-right corner: oh=3, ow=3, kr=2, kc=2 → ih=4, iw=4  INVALID
        check_query(4,4,1,1, 3,3, 0, 2,2,  0,0, "botright_corner", exp_v);
        tc_errors += exp_v;

        // Top-right corner: oh=0, ow=3, kr=0, kc=2 → ih=-1, iw=4  INVALID
        check_query(4,4,1,1, 0,3, 0, 0,2,  0,0, "topright_corner", exp_v);
        tc_errors += exp_v;

        // Bottom-left corner: oh=3, ow=0, kr=2, kc=0 → ih=4, iw=-1  INVALID
        check_query(4,4,1,1, 3,0, 0, 2,0,  0,0, "botleft_corner", exp_v);
        tc_errors += exp_v;

        total_errors += tc_errors;
        if (tc_errors == 0) $display("[Group2_BoundaryCases] PASS — all 8 boundary checks correct");
        else                $display("[Group2_BoundaryCases] FAIL — %0d mismatches", tc_errors);

        // =================================================================
        // Group 3: Address formula spot-checks
        //   H_in=W_in=4, stride=1, padding=1
        //   Hand-computed expected addresses for interior pixels.
        // =================================================================
        $display("\n=== Group 3: Address formula spot-checks (H=W=4, s=1, p=1) ===");
        H_in = 8'd4; W_in = 8'd4; stride = 4'd1; padding = 4'd1;
        tc_errors = 0;

        // oh=0, ow=0, ic=0, kr=1, kc=1: ih=0, iw=0 → addr = (0*4+0)*4+0 = 0
        check_query(4,4,1,1, 0,0, 0, 1,1,  0,1, "ic0_oh0_ow0_centre", exp_v);
        tc_errors += exp_v;

        // oh=0, ow=0, ic=0, kr=1, kc=2: ih=0, iw=1 → addr = 1
        check_query(4,4,1,1, 0,0, 0, 1,2,  1,1, "ic0_oh0_ow0_right", exp_v);
        tc_errors += exp_v;

        // oh=1, ow=1, ic=0, kr=1, kc=1: ih=1, iw=1 → addr = (0*4+1)*4+1 = 5
        check_query(4,4,1,1, 1,1, 0, 1,1,  5,1, "ic0_oh1_ow1_centre", exp_v);
        tc_errors += exp_v;

        // oh=2, ow=3, ic=0, kr=0, kc=0: ih=1, iw=2 → addr = (0*4+1)*4+2 = 6
        check_query(4,4,1,1, 2,3, 0, 0,0,  6,1, "ic0_oh2_ow3_kr0_kc0", exp_v);
        tc_errors += exp_v;

        // oh=3, ow=3, ic=0, kr=1, kc=1: ih=3, iw=3 → addr = (0*4+3)*4+3 = 15
        check_query(4,4,1,1, 3,3, 0, 1,1,  15,1, "ic0_oh3_ow3_centre", exp_v);
        tc_errors += exp_v;

        // Channel offset: ic=1, oh=0, ow=0, kr=1, kc=1: ih=0, iw=0
        //   addr = (1*4+0)*4+0 = 16
        check_query(4,4,1,1, 0,0, 1, 1,1,  16,1, "ic1_oh0_ow0_centre", exp_v);
        tc_errors += exp_v;

        // Channel offset: ic=1, oh=2, ow=2, kr=2, kc=2: ih=3, iw=3
        //   addr = (1*4+3)*4+3 = 31
        check_query(4,4,1,1, 2,2, 1, 2,2,  31,1, "ic1_oh2_ow2_br", exp_v);
        tc_errors += exp_v;

        total_errors += tc_errors;
        if (tc_errors == 0) $display("[Group3_SpotChecks] PASS — all 7 address checks correct");
        else                $display("[Group3_SpotChecks] FAIL — %0d mismatches", tc_errors);

        // =================================================================
        // Group 4: Stride=2 operation
        //   H_in=W_in=8, stride=2, padding=1
        //   H_out = W_out = (8 + 2*1 - 3)/2 + 1 = 4
        // =================================================================
        $display("\n=== Group 4: Stride=2 (H=W=8, s=2, p=1) ===");
        H_in = 8'd8; W_in = 8'd8; stride = 4'd2; padding = 4'd1;
        tc_errors = 0;

        // oh=0, ow=0, kr=0, kc=0: ih=-1, iw=-1  INVALID (padding region)
        check_query(8,8,2,1, 0,0, 0, 0,0,  0,0, "s2_topleft_pad", exp_v);
        tc_errors += exp_v;

        // oh=0, ow=0, kr=1, kc=1: ih=0, iw=0  → addr=0  VALID
        check_query(8,8,2,1, 0,0, 0, 1,1,  0,1, "s2_oh0_ow0_centre", exp_v);
        tc_errors += exp_v;

        // oh=1, ow=1, kr=1, kc=1: ih=2, iw=2  → addr=(0*8+2)*8+2=18  VALID
        check_query(8,8,2,1, 1,1, 0, 1,1,  18,1, "s2_oh1_ow1_centre", exp_v);
        tc_errors += exp_v;

        // oh=3, ow=3, kr=2, kc=2: ih=7, iw=7  → addr=(0*8+7)*8+7=63  VALID
        check_query(8,8,2,1, 3,3, 0, 2,2,  63,1, "s2_oh3_ow3_br", exp_v);
        tc_errors += exp_v;

        // oh=3, ow=3, kr=2, kc=2 with ic=1: addr=(1*8+7)*8+7=127  VALID
        check_query(8,8,2,1, 3,3, 1, 2,2,  127,1, "s2_ic1_oh3_ow3_br", exp_v);
        tc_errors += exp_v;

        // oh=3, ow=2, kr=2, kc=2: ih=7, iw=5+1=6?
        //   ih = 3*2 + 2 - 1 = 7, iw = 2*2 + 2 - 1 = 5  → addr=(0*8+7)*8+5=61
        check_query(8,8,2,1, 3,2, 0, 2,2,  61,1, "s2_oh3_ow2_br", exp_v);
        tc_errors += exp_v;

        total_errors += tc_errors;
        if (tc_errors == 0) $display("[Group4_Stride2] PASS — all 6 stride-2 checks correct");
        else                $display("[Group4_Stride2] FAIL — %0d mismatches", tc_errors);

        // =================================================================
        // Group 5: Exhaustive sweep — H=W=4, stride=1, padding=1, ic 0..1
        //   288 cases total (4×4 output pixels × 9 kernel pos × 2 channels)
        //   All checked against ref_addr golden model.
        // =================================================================
        $display("\n=== Group 5: Exhaustive sweep H=W=4, s=1, p=1, ic=0..1 (288 cases) ===");
        H_in = 8'd4; W_in = 8'd4; stride = 4'd1; padding = 4'd1;
        tc_errors = 0;
        begin
            int sweep_errors;
            sweep_errors = 0;
            for (int o_h = 0; o_h < 4; o_h++) begin
                for (int o_w = 0; o_w < 4; o_w++) begin
                    for (int c = 0; c < 2; c++) begin
                        for (int k_r = 0; k_r < 3; k_r++) begin
                            for (int k_c = 0; k_c < 3; k_c++) begin
                                ref_addr(4, 4, 1, 1, o_h, o_w, c, k_r, k_c, exp_a, exp_v);

                                // Drive DUT
                                oh = 8'(o_h); ow = 8'(o_w);
                                ic = 8'(c);
                                kr = 2'(k_r); kc = 2'(k_c);
                                query_valid = 1'b1;
                                #1;

                                if (int'(addr_valid) !== exp_v ||
                                    (exp_v && int'(sram_addr) !== exp_a)) begin
                                    $display("LOG: %0t : ERROR : tb_conv_addr_gen : dut.sram_addr/addr_valid : expected_value: valid=%0d addr=%0d actual_value: valid=%0d addr=%0d  [sweep oh=%0d ow=%0d ic=%0d kr=%0d kc=%0d]",
                                             $time, exp_v, exp_a,
                                             int'(addr_valid), int'(sram_addr),
                                             o_h, o_w, c, k_r, k_c);
                                    sweep_errors++;
                                end
                            end
                        end
                    end
                end
            end
            tc_errors = sweep_errors;
            if (sweep_errors == 0)
                $display("[Group5_ExhaustiveH4] PASS — all 288 cases match golden");
            else
                $display("[Group5_ExhaustiveH4] FAIL — %0d/288 mismatches", sweep_errors);
        end
        total_errors += tc_errors;

        // =================================================================
        // Group 6: Exhaustive sweep — H=W=8, stride=1, padding=1, ic=0
        //   576 cases (8×8 output pixels × 9 kernel positions)
        // =================================================================
        $display("\n=== Group 6: Exhaustive sweep H=W=8, s=1, p=1, ic=0 (576 cases) ===");
        H_in = 8'd8; W_in = 8'd8; stride = 4'd1; padding = 4'd1;
        tc_errors = 0;
        begin
            int sweep_errors;
            sweep_errors = 0;
            for (int o_h = 0; o_h < 8; o_h++) begin
                for (int o_w = 0; o_w < 8; o_w++) begin
                    for (int k_r = 0; k_r < 3; k_r++) begin
                        for (int k_c = 0; k_c < 3; k_c++) begin
                            ref_addr(8, 8, 1, 1, o_h, o_w, 0, k_r, k_c, exp_a, exp_v);

                            oh = 8'(o_h); ow = 8'(o_w);
                            ic = 8'd0;
                            kr = 2'(k_r); kc = 2'(k_c);
                            query_valid = 1'b1;
                            #1;

                            if (int'(addr_valid) !== exp_v ||
                                (exp_v && int'(sram_addr) !== exp_a)) begin
                                $display("LOG: %0t : ERROR : tb_conv_addr_gen : dut.sram_addr/addr_valid : expected_value: valid=%0d addr=%0d actual_value: valid=%0d addr=%0d  [sweep oh=%0d ow=%0d kr=%0d kc=%0d]",
                                         $time, exp_v, exp_a,
                                         int'(addr_valid), int'(sram_addr),
                                         o_h, o_w, k_r, k_c);
                                sweep_errors++;
                            end
                        end
                    end
                end
            end
            tc_errors = sweep_errors;
            if (sweep_errors == 0)
                $display("[Group6_ExhaustiveH8] PASS — all 576 cases match golden");
            else
                $display("[Group6_ExhaustiveH8] FAIL — %0d/576 mismatches", sweep_errors);
        end
        total_errors += tc_errors;

        // =================================================================
        // Final result
        // =================================================================
        query_valid = 1'b0;
        #10;

        if (total_errors == 0) begin
            $display("\nTEST PASSED");
            $finish;
        end else begin
            $display("\nERROR");
            $error("tb_conv_addr_gen: %0d total mismatches", total_errors);
            $fatal(1, "tb_conv_addr_gen FAILED");
        end
    end

    // Watchdog
    initial begin
        #500_000;
        $display("LOG: %0t : ERROR : tb_conv_addr_gen : tb.watchdog : expected_value: completion actual_value: timeout",
                 $time);
        $display("ERROR");
        $fatal(1, "Watchdog timeout");
    end

endmodule
