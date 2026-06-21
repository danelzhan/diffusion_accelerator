// =============================================================================
// matmul_controller.sv  —  4×4 Matrix Multiplication Controller
// =============================================================================
//
// Drives a fixed 4×4 × 4×4 INT8 matrix multiplication through the
// systolic_array_4x4 and manages all SRAM traffic.
//
// Operation
// ---------
//   Given matrices A (4×4, INT8) and B (4×4, INT8) pre-loaded into their
//   respective SRAMs, the controller:
//     1. Clears the systolic array accumulators.
//     2. Loads all 16 A-values and 16 B-values into local registers
//        (sequential SRAM reads, two SRAMs in parallel).
//     3. Feeds K=4 column-slices of A and row-slices of B into the array,
//        asserting valid_in for exactly 4 cycles.
//     4. Waits 6 drain cycles for the pipeline to flush through PE[3][3].
//     5. Captures the 4×4 INT32 result from c_out and writes it to the
//        output SRAM sequentially (16 writes).
//     6. Pulses done for one cycle.
//
// SRAM Layout  (all row-major, zero-indexed)
// ------------------------------------------
//   A SRAM (INT8,  depth ≥ 16):  A[i][k] → address  i*4 + k
//   B SRAM (INT8,  depth ≥ 16):  B[k][j] → address  k*4 + j
//   C SRAM (INT32, depth ≥ 16):  C[i][j] → address  i*4 + j
//
// FSM States
// ----------
//   IDLE      : wait for start pulse
//   CLEAR     : broadcast clear=1 to systolic array for one cycle
//   LOAD      : read A[0..15] and B[0..15] from SRAM into a_reg / b_reg
//               (17 cycles: 16 address cycles + 1 extra for SRAM read latency)
//   FEED      : stream K=4 slices into the array with valid_in=1
//   DRAIN     : hold valid_in=0 for 6 cycles (pipeline flush latency)
//   WRITE_OUT : write 16 INT32 results to C SRAM (16 cycles)
//   DONE      : assert done for one cycle, return to IDLE
//
// SRAM Read Latency Handling (LOAD phase)
// ----------------------------------------
//   load_cnt drives raddr each cycle.  Because sram_model has 1-cycle read
//   latency, rdata at cycle (load_cnt = N) reflects raddr from cycle N-1.
//   Capture schedule:
//     load_cnt = 0  → present raddr = 0,   no capture
//     load_cnt = 1  → present raddr = 1,   capture reg[0]  = rdata
//     ...
//     load_cnt = 15 → present raddr = 15,  capture reg[14] = rdata
//     load_cnt = 16 → no new raddr (re=0), capture reg[15] = rdata
//   After 17 cycles (load_cnt 0→16), a_reg[0..15] and b_reg[0..15] are full.
//
// Systolic Array Feed (FEED phase)
// ---------------------------------
//   Each cycle k = feed_cnt (0..3):
//     sa_a_in[i] = a_reg[i*4 + k]   ← column-k slice of A
//     sa_b_in[j] = b_reg[k*4 + j]   ← row-k slice of B
//   The systolic array internally skews these inputs so each PE[i][j]
//   accumulates A[i][k] × B[k][j] at the correct time.
//
// Total cycle budget
// ------------------
//   CLEAR(1) + LOAD(17) + FEED(4) + DRAIN(6) + WRITE_OUT(16) + DONE(1) = 45
//
// Ports
// -----
//   clock, reset  : standard
//   start         : pulse high for one cycle to begin operation
//   done          : pulses high for one cycle when C SRAM write is complete
//
//   a_re / a_raddr / a_rdata  : A SRAM read interface
//   b_re / b_raddr / b_rdata  : B SRAM read interface
//   c_we / c_waddr / c_wdata  : C SRAM write interface
//
//   sa_clear    : broadcast clear to all PEs
//   sa_valid_in : data-valid qualifier to systolic array
//   sa_a_in[4]  : activation row inputs (INT8)
//   sa_b_in[4]  : weight column inputs (INT8)
//   sa_c_out[4][4] : INT32 accumulator outputs (read combinatorially)
//
// =============================================================================

module matmul_controller (
    input  logic               clock,
    input  logic               reset,
    input  logic               start,
    output logic               done,

    // ── A SRAM read interface (INT8, 16 entries) ─────────────────────────────
    output logic               a_re,
    output logic [3:0]         a_raddr,
    input  logic signed [7:0]  a_rdata,

    // ── B SRAM read interface (INT8, 16 entries) ─────────────────────────────
    output logic               b_re,
    output logic [3:0]         b_raddr,
    input  logic signed [7:0]  b_rdata,

    // ── C SRAM write interface (INT32, 16 entries) ───────────────────────────
    output logic               c_we,
    output logic [3:0]         c_waddr,
    output logic signed [31:0] c_wdata,

    // ── Systolic array interface ──────────────────────────────────────────────
    output logic               sa_clear,
    output logic               sa_valid_in,
    output logic signed [7:0]  sa_a_in [4],
    output logic signed [7:0]  sa_b_in [4],
    input  logic signed [31:0] sa_c_out [4][4]
);

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE      = 3'd0,
        CLEAR     = 3'd1,
        LOAD      = 3'd2,
        FEED      = 3'd3,
        DRAIN     = 3'd4,
        WRITE_OUT = 3'd5,
        DONE      = 3'd6
    } state_t;

    state_t state, next_state;

    // =========================================================================
    // Counters
    // =========================================================================
    logic [4:0] load_cnt;   // 0..16  (17 steps for SRAM latency)
    logic [1:0] feed_cnt;   // 0..3   (K = 4 accumulation steps)
    logic [2:0] drain_cnt;  // 0..5   (6 drain cycles)
    logic [3:0] write_cnt;  // 0..15  (16 output writes)

    // =========================================================================
    // Local matrix registers (loaded from SRAM before compute)
    // =========================================================================
    logic signed [7:0] a_reg [16]; // a_reg[i*4 + k] = A[i][k]
    logic signed [7:0] b_reg [16]; // b_reg[k*4 + j] = B[k][j]

    // =========================================================================
    // State register  (synchronous reset)
    // =========================================================================
    always_ff @(posedge clock) begin
        if (reset)
            state <= IDLE;
        else
            state <= next_state;
    end

    // =========================================================================
    // Next-state logic  (combinational)
    // =========================================================================
    always_comb begin
        next_state = state;
        case (state)
            IDLE:      next_state = start            ? CLEAR     : IDLE;
            CLEAR:     next_state = LOAD;
            LOAD:      next_state = (load_cnt  == 5'd16) ? FEED      : LOAD;
            FEED:      next_state = (feed_cnt  == 2'd3)  ? DRAIN     : FEED;
            DRAIN:     next_state = (drain_cnt == 3'd5)  ? WRITE_OUT : DRAIN;
            WRITE_OUT: next_state = (write_cnt == 4'd15) ? DONE      : WRITE_OUT;
            DONE:      next_state = IDLE;
            default:   next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Counter updates  (synchronous, reset to 0 in non-owning states)
    // =========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            load_cnt  <= '0;
            feed_cnt  <= '0;
            drain_cnt <= '0;
            write_cnt <= '0;
        end else begin
            case (state)
                LOAD:      load_cnt  <= load_cnt  + 5'd1;
                FEED:      feed_cnt  <= feed_cnt  + 2'd1;
                DRAIN:     drain_cnt <= drain_cnt + 3'd1;
                WRITE_OUT: write_cnt <= write_cnt + 4'd1;
                default: begin
                    load_cnt  <= '0;
                    feed_cnt  <= '0;
                    drain_cnt <= '0;
                    write_cnt <= '0;
                end
            endcase
        end
    end

    // =========================================================================
    // SRAM load: capture A and B values into local registers
    //
    // rdata at load_cnt=N contains mem[N-1] (1-cycle latency), so we
    // write a_reg[N-1] when load_cnt >= 1.  The final entry (index 15) is
    // captured at load_cnt=16 (the extra cycle after the last address).
    // =========================================================================
    always_ff @(posedge clock) begin
        if (state == LOAD && load_cnt >= 5'd1) begin
            a_reg[load_cnt - 5'd1] <= a_rdata;
            b_reg[load_cnt - 5'd1] <= b_rdata;
        end
    end

    // =========================================================================
    // A / B SRAM read drives  (combinational)
    //   Drive re and raddr only during LOAD while load_cnt < 16.
    //   At load_cnt=16, re drops low (last data is already in flight).
    // =========================================================================
    always_comb begin
        a_re    = (state == LOAD) && (load_cnt < 5'd16);
        a_raddr = load_cnt[3:0];
        b_re    = (state == LOAD) && (load_cnt < 5'd16);
        b_raddr = load_cnt[3:0];
    end

    // =========================================================================
    // Systolic array drives  (combinational)
    // =========================================================================
    always_comb begin
        sa_clear     = (state == CLEAR);
        sa_valid_in  = (state == FEED);

        // Default: drive zeros so no spurious accumulation occurs
        for (int p = 0; p < 4; p++) begin
            sa_a_in[p] = 8'sd0;
            sa_b_in[p] = 8'sd0;
        end

        if (state == FEED) begin
            // Column-k slice of A:  A[i][k] = a_reg[i*4 + feed_cnt]
            for (int i = 0; i < 4; i++)
                sa_a_in[i] = a_reg[i * 4 + int'(feed_cnt)];

            // Row-k slice of B:    B[k][j] = b_reg[feed_cnt*4 + j]
            for (int j = 0; j < 4; j++)
                sa_b_in[j] = b_reg[int'(feed_cnt) * 4 + j];
        end
    end

    // =========================================================================
    // C SRAM write  (combinational drive, registered inside sram_model)
    //   Sequentially write sa_c_out[row][col] in row-major order.
    //   write_cnt maps:  row = write_cnt[3:2],  col = write_cnt[1:0]
    // =========================================================================
    always_comb begin
        c_we    = (state == WRITE_OUT);
        c_waddr = write_cnt;
        c_wdata = sa_c_out[write_cnt[3:2]][write_cnt[1:0]];
    end

    // =========================================================================
    // Done flag
    // =========================================================================
    assign done = (state == DONE);

endmodule
