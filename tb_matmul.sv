// =============================================================================
// tb_matmul.sv  —  Testbench for the Phase-1 Matmul Path
// =============================================================================
//
// Instantiates and exercises the full matmul datapath:
//
//   tb_matmul
//     ├── u_a_sram  : sram_model #(.DEPTH(16),.DWIDTH(8))   — INT8  A matrix
//     ├── u_b_sram  : sram_model #(.DEPTH(16),.DWIDTH(8))   — INT8  B matrix
//     ├── u_c_sram  : sram_model #(.DEPTH(16),.DWIDTH(32))  — INT32 C output
//     ├── u_ctrl    : matmul_controller
//     └── u_sa      : systolic_array_4x4  (16× pe.sv)
//
// Test cases
// ----------
//   1. All-ones:        A=1,B=1        → C[i][j] = 4  (4 products of 1×1)
//   2. Identity:        A=1..16, B=I   → C = A         (pass-through check)
//   3. Negative values: A=-1,  B=2     → C[i][j] = -8  (signed arithmetic)
//   4. Mixed-sign:      hand-picked    → C via ref model (corner case stress)
//
// Verification method
// -------------------
//   A software golden model (ref_matmul function) computes the expected INT32
//   output tile.  After each run, the C SRAM is read back and every element is
//   compared against the golden value.  Any mismatch is logged at ERROR level
//   and counted.  Final pass/fail is based on total error count across all tests.
//
// Timing conventions
// ------------------
//   • Clock period: 10 ns (posedge at 5, 15, 25, ...)
//   • All testbench signal drives happen at (@posedge clock) + #1 to ensure
//     the driven value is sampled by RTL flip-flops on the NEXT rising edge,
//     avoiding race conditions with the RTL always_ff blocks.
//   • SRAM read latency: 1 cycle — address presented at T, rdata valid at T+1.
//     The check_result task accounts for this by issuing an extra clock.
//
// Expected output (lines starting with LOG:)
// -------------------------------------------
//   LOG: <time> : INFO  : tb_matmul : dut.u_c_sram.mem[N] : expected_value: X actual_value: X
//   LOG: <time> : ERROR : tb_matmul : dut.u_c_sram.mem[N] : expected_value: X actual_value: Y
//
// =============================================================================

`timescale 1ns/1ps

module tb_matmul;

    // =========================================================================
    // Clock
    // =========================================================================
    localparam CLK_HALF = 5; // 10 ns period
    logic clock, reset;

    initial  clock = 1'b0;
    always  #CLK_HALF clock = ~clock;

    // =========================================================================
    // Testbench interface signals
    // =========================================================================

    // Controller
    logic start, done;

    // Testbench → A SRAM (write)
    logic       tb_a_we;
    logic [3:0] tb_a_waddr;
    logic [7:0] tb_a_wdata;

    // Controller ← A SRAM (read)
    logic       ctrl_a_re;
    logic [3:0] ctrl_a_raddr;
    logic [7:0] ctrl_a_rdata;

    // Testbench → B SRAM (write)
    logic       tb_b_we;
    logic [3:0] tb_b_waddr;
    logic [7:0] tb_b_wdata;

    // Controller ← B SRAM (read)
    logic       ctrl_b_re;
    logic [3:0] ctrl_b_raddr;
    logic [7:0] ctrl_b_rdata;

    // Controller → C SRAM (write)
    logic        ctrl_c_we;
    logic [3:0]  ctrl_c_waddr;
    logic [31:0] ctrl_c_wdata;

    // Testbench ← C SRAM (read)
    logic        tb_c_re;
    logic [3:0]  tb_c_raddr;
    logic [31:0] tb_c_rdata;

    // Controller ↔ Systolic array wires
    logic               sa_clear;
    logic               sa_valid_in;
    logic               sa_valid_out;
    logic signed [7:0]  sa_a_in  [4];
    logic signed [7:0]  sa_b_in  [4];
    logic signed [31:0] sa_c_out [4][4];

    // =========================================================================
    // DUT instantiation
    // =========================================================================

    // A matrix SRAM (INT8, 16 entries: A[i][k] at addr i*4+k)
    sram_model #(.DEPTH(16), .DWIDTH(8)) u_a_sram (
        .clock (clock),        .reset (reset),
        .we    (tb_a_we),      .waddr (tb_a_waddr),    .wdata (tb_a_wdata),
        .re    (ctrl_a_re),    .raddr (ctrl_a_raddr),  .rdata (ctrl_a_rdata)
    );

    // B matrix SRAM (INT8, 16 entries: B[k][j] at addr k*4+j)
    sram_model #(.DEPTH(16), .DWIDTH(8)) u_b_sram (
        .clock (clock),        .reset (reset),
        .we    (tb_b_we),      .waddr (tb_b_waddr),    .wdata (tb_b_wdata),
        .re    (ctrl_b_re),    .raddr (ctrl_b_raddr),  .rdata (ctrl_b_rdata)
    );

    // C matrix SRAM (INT32, 16 entries: C[i][j] at addr i*4+j)
    sram_model #(.DEPTH(16), .DWIDTH(32)) u_c_sram (
        .clock (clock),        .reset (reset),
        .we    (ctrl_c_we),    .waddr (ctrl_c_waddr),  .wdata (ctrl_c_wdata),
        .re    (tb_c_re),      .raddr (tb_c_raddr),    .rdata (tb_c_rdata)
    );

    // Matrix multiplication controller
    matmul_controller u_ctrl (
        .clock       (clock),         .reset       (reset),
        .start       (start),         .done        (done),
        .a_re        (ctrl_a_re),     .a_raddr     (ctrl_a_raddr),  .a_rdata (ctrl_a_rdata),
        .b_re        (ctrl_b_re),     .b_raddr     (ctrl_b_raddr),  .b_rdata (ctrl_b_rdata),
        .c_we        (ctrl_c_we),     .c_waddr     (ctrl_c_waddr),  .c_wdata (ctrl_c_wdata),
        .sa_clear    (sa_clear),      .sa_valid_in (sa_valid_in),
        .sa_a_in     (sa_a_in),       .sa_b_in     (sa_b_in),
        .sa_c_out    (sa_c_out)
    );

    // 4×4 output-stationary systolic array (16× pe.sv)
    systolic_array_4x4 u_sa (
        .clock    (clock),       .reset     (reset),
        .clear    (sa_clear),    .valid_in  (sa_valid_in),
        .a_in     (sa_a_in),     .b_in      (sa_b_in),
        .c_out    (sa_c_out),    .valid_out (sa_valid_out)
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
    //   Computes C = A × B in INT32 arithmetic, matching the hardware behaviour.
    //   sign-extends each INT8 input before multiplying (identical to the PE).
    // =========================================================================
    function automatic void ref_matmul(
        input  logic signed [7:0]  a [4][4],
        input  logic signed [7:0]  b [4][4],
        output logic signed [31:0] c [4][4]
    );
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                c[i][j] = 32'sd0;
                for (int k = 0; k < 4; k++)
                    c[i][j] += 32'(signed'(a[i][k])) * 32'(signed'(b[k][j]));
            end
        end
    endfunction

    // =========================================================================
    // Tasks
    // =========================================================================

    // ------------------------------------------------------------------
    // load_a_sram: write a 4×4 INT8 matrix into the A SRAM.
    //   Layout: A[i][k] → address i*4+k  (row-major)
    //   One write per clock cycle; 16 cycles total.
    // ------------------------------------------------------------------
    task automatic load_a_sram(input logic signed [7:0] mat [4][4]);
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                @(posedge clock); #1;
                tb_a_we    = 1'b1;
                tb_a_waddr = 4'(i * 4 + j);
                tb_a_wdata = mat[i][j];
            end
        end
        @(posedge clock); #1;
        tb_a_we = 1'b0;
    endtask

    // ------------------------------------------------------------------
    // load_b_sram: write a 4×4 INT8 matrix into the B SRAM.
    //   Layout: B[k][j] → address k*4+j  (row-major)
    // ------------------------------------------------------------------
    task automatic load_b_sram(input logic signed [7:0] mat [4][4]);
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                @(posedge clock); #1;
                tb_b_we    = 1'b1;
                tb_b_waddr = 4'(i * 4 + j);
                tb_b_wdata = mat[i][j];
            end
        end
        @(posedge clock); #1;
        tb_b_we = 1'b0;
    endtask

    // ------------------------------------------------------------------
    // run_and_wait: pulse start for one cycle, then poll done.
    //   Maximum wait: 300 cycles (well above the 45-cycle FSM budget).
    // ------------------------------------------------------------------
    task automatic run_and_wait();
        int timeout;
        @(posedge clock); #1;
        start = 1'b1;
        @(posedge clock); #1;
        start   = 1'b0;
        timeout = 0;
        while (!done) begin
            @(posedge clock); #1;
            timeout++;
            if (timeout > 300) begin
                $display("LOG: %0t : ERROR : tb_matmul : tb.done : expected_value: 1 actual_value: 0",
                         $time);
                $display("ERROR");
                $fatal(1, "Timeout: done not received within 300 cycles of start");
            end
        end
        // Let done pulse complete before reading results
        @(posedge clock); #1;
    endtask

    // ------------------------------------------------------------------
    // check_result: read all 16 C SRAM entries, compare vs golden.
    //   SRAM read latency = 1 cycle: present raddr at T, rdata valid at T+1.
    //   Returns number of mismatches in err_count.
    // ------------------------------------------------------------------
    task automatic check_result(
        input  logic signed [31:0] golden [4][4],
        input  string              label,
        output int                 err_count
    );
        logic signed [31:0] actual [4][4];
        err_count = 0;

        // Read each entry: drive addr → wait one cycle → sample rdata
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                @(posedge clock); #1;
                tb_c_re    = 1'b1;
                tb_c_raddr = 4'(i * 4 + j);
                @(posedge clock); #1;          // rdata = mem[addr] is now stable
                actual[i][j] = signed'(tb_c_rdata);
            end
        end
        @(posedge clock); #1;
        tb_c_re = 1'b0;

        // Compare every element and log
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                if (actual[i][j] !== golden[i][j]) begin
                    $display("LOG: %0t : ERROR : tb_matmul : dut.u_c_sram.mem[%0d] : expected_value: %0d actual_value: %0d",
                             $time, i*4+j, golden[i][j], actual[i][j]);
                    err_count++;
                end else begin
                    $display("LOG: %0t : INFO  : tb_matmul : dut.u_c_sram.mem[%0d] : expected_value: %0d actual_value: %0d",
                             $time, i*4+j, golden[i][j], actual[i][j]);
                end
            end
        end

        if (err_count == 0)
            $display("[%s] PASS — all 16 outputs match golden", label);
        else
            $display("[%s] FAIL — %0d/16 outputs mismatch", label, err_count);
    endtask

    // =========================================================================
    // Main stimulus block
    // =========================================================================
    int total_errors;
    int tc_errors;

    logic signed [7:0]  A [4][4];
    logic signed [7:0]  B [4][4];
    logic signed [31:0] golden [4][4];

    initial begin
        $display("TEST START");
        total_errors = 0;

        // Initialise all driven signals to safe defaults
        reset      = 1'b1;
        start      = 1'b0;
        tb_a_we    = 1'b0;   tb_a_waddr = 4'd0;   tb_a_wdata = 8'd0;
        tb_b_we    = 1'b0;   tb_b_waddr = 4'd0;   tb_b_wdata = 8'd0;
        tb_c_re    = 1'b0;   tb_c_raddr = 4'd0;

        // Hold reset for 4 cycles
        repeat (4) @(posedge clock);
        #1; reset = 1'b0;
        repeat (2) @(posedge clock);

        // =================================================================
        // Test 1: All-ones matrices
        //   A[i][j] = 1, B[i][j] = 1
        //   Expected: C[i][j] = 4  (dot product of four 1×1 terms)
        // =================================================================
        $display("\n=== Test 1: A=ones, B=ones  (C[i][j] expected = 4) ===");
        foreach (A[i,j]) A[i][j] = 8'sd1;
        foreach (B[i,j]) B[i][j] = 8'sd1;
        ref_matmul(A, B, golden);

        load_a_sram(A);
        load_b_sram(B);
        run_and_wait();
        check_result(golden, "Test1_AllOnes", tc_errors);
        total_errors += tc_errors;

        // =================================================================
        // Test 2: Sequential A × Identity B  →  C = A
        //   A[i][j] = i*4+j+1  (values 1..16, all fit in INT8)
        //   B        = 4×4 identity matrix
        //   Expected: C = A  (tests correct row/column routing)
        // =================================================================
        $display("\n=== Test 2: A=sequential(1..16), B=identity  (C expected = A) ===");
        foreach (A[i,j]) A[i][j] = 8'(i * 4 + j + 1);
        foreach (B[i,j]) B[i][j] = (i == j) ? 8'sd1 : 8'sd0;
        ref_matmul(A, B, golden);

        load_a_sram(A);
        load_b_sram(B);
        run_and_wait();
        check_result(golden, "Test2_AxIdentity", tc_errors);
        total_errors += tc_errors;

        // =================================================================
        // Test 3: Signed arithmetic — all negative A, positive B
        //   A[i][j] = -1, B[i][j] = 2
        //   Expected: C[i][j] = 4 × (-1×2) = -8
        //   Tests sign-extension through the full systolic pipeline.
        // =================================================================
        $display("\n=== Test 3: A=(-1), B=2  (C[i][j] expected = -8) ===");
        foreach (A[i,j]) A[i][j] = -8'sd1;
        foreach (B[i,j]) B[i][j] =  8'sd2;
        ref_matmul(A, B, golden);

        load_a_sram(A);
        load_b_sram(B);
        run_and_wait();
        check_result(golden, "Test3_NegativeValues", tc_errors);
        total_errors += tc_errors;

        // =================================================================
        // Test 4: Mixed-sign arbitrary matrices
        //   Stresses positive × negative, negative × positive, and
        //   negative × negative combinations in the same tile.
        //   Golden is computed by ref_matmul — no hand-verification needed.
        // =================================================================
        $display("\n=== Test 4: Mixed-sign arbitrary matrices ===");
        A[0] = '{  8'sd3, -8'sd1,  8'sd5, -8'sd2 };
        A[1] = '{ -8'sd4,  8'sd6, -8'sd3,  8'sd1 };
        A[2] = '{  8'sd2, -8'sd5,  8'sd7, -8'sd4 };
        A[3] = '{ -8'sd1,  8'sd3, -8'sd6,  8'sd8 };
        B[0] = '{  8'sd1, -8'sd2,  8'sd3, -8'sd4 };
        B[1] = '{ -8'sd3,  8'sd5, -8'sd1,  8'sd2 };
        B[2] = '{  8'sd2, -8'sd4,  8'sd6, -8'sd3 };
        B[3] = '{ -8'sd5,  8'sd1, -8'sd2,  8'sd7 };
        ref_matmul(A, B, golden);

        load_a_sram(A);
        load_b_sram(B);
        run_and_wait();
        check_result(golden, "Test4_MixedSign", tc_errors);
        total_errors += tc_errors;

        // =================================================================
        // Final result
        // =================================================================
        repeat (4) @(posedge clock);

        if (total_errors == 0) begin
            $display("\nTEST PASSED");
            $finish;
        end else begin
            $display("\nERROR");
            $error("tb_matmul: %0d output mismatch(es) across 4 test cases",
                   total_errors);
            $fatal(1, "tb_matmul FAILED");
        end
    end

    // =========================================================================
    // Global watchdog — prevents simulator from hanging on deadlock/livelock
    // =========================================================================
    initial begin
        #500_000; // 500 us — far beyond expected ~200 cycle runtime
        $display("LOG: %0t : ERROR : tb_matmul : tb.watchdog : expected_value: simulation_complete actual_value: timeout_exceeded",
                 $time);
        $display("ERROR");
        $fatal(1, "Watchdog: simulation did not complete within 500 us");
    end

endmodule
