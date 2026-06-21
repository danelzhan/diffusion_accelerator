// =============================================================================
// systolic_array_4x4.sv  —  4×4 Output-Stationary Systolic Array
// =============================================================================
//
// Architecture overview
// ----------------------
//   • 4×4 grid of pe.sv instances.
//   • A (activation) data streams left → right across rows.
//   • B (weight) data streams top → bottom down columns.
//   • Each PE accumulates acc += a * b while valid_in is asserted.
//   • c_out[i][j] is PE[i][j]'s accumulator — stable after computation
//     completes (controller manages timing).
//
// Input skewing (diagonal delay)
// --------------------------------
//   For C = A×B with K inner-product steps, PE[i][j] must receive
//   A[i][k] and B[k][j] at the same clock cycle.
//
//   Without skewing:
//     A[i][k] takes j extra cycles to reach PE[i][j] (j PE hops right).
//     B[k][j] takes i extra cycles to reach PE[i][j] (i PE hops down).
//
//   Solution: pre-skew inputs inside this module:
//     Row i  → delay A input by i cycles   (shift-register bank)
//     Col j  → delay B input by j cycles   (shift-register bank)
//
//   With skewing, A[i][k] enters at cycle (k + i) and arrives at PE[i][j]
//   at cycle (k + i + j).  B[k][j] enters at cycle (k + j) and arrives
//   at PE[i][j] at cycle (k + j + i).  They meet at (k + i + j). ✓
//
//   Valid signal skewing:
//     valid for row i is delayed by i cycles (travels with A horizontally).
//     PE[i][j].valid_in is therefore delayed by (i + j) from valid_in,
//     matching the data alignment above.
//
// Latency
// -------
//   Total cycles for a K-step matmul = K + 6  (pipeline fill = 3 + drain = 3).
//   PE[3][3].valid_out (= valid_out of this module) transitions after the
//   last valid token drains through the corner PE.
//
// Ports
// -----
//   clock      : system clock
//   reset      : synchronous active-high reset
//   clear      : broadcast clear to all PE accumulators
//   a_in[4]    : one INT8 activation per row, fed each clock cycle
//   b_in[4]    : one INT8 weight per column, fed each clock cycle
//   valid_in   : high for K consecutive cycles during a computation
//   c_out[4][4]: INT32 accumulator outputs, PE[row][col]
//   valid_out  : valid propagated to corner PE[3][3]; indicates last data
//                has reached the bottom-right of the array
//
// =============================================================================

module systolic_array_4x4 (
    input  logic               clock,
    input  logic               reset,
    input  logic               clear,

    // Activation row inputs (A matrix column-slice per cycle)
    input  logic signed [7:0]  a_in [4],
    // Weight column inputs (B matrix row-slice per cycle)
    input  logic signed [7:0]  b_in [4],
    input  logic               valid_in,

    // Accumulated output tile
    output logic signed [31:0] c_out [4][4],
    // Corner valid — set when last data token has passed through PE[3][3]
    output logic               valid_out
);

    // =========================================================================
    // 1. Input skew shift-register banks
    //    a_stage[i][s]: A row-i data at pipeline stage s (0 = direct input)
    //    b_stage[j][s]: B col-j data at pipeline stage s (0 = direct input)
    //    v_stage[s]   : valid at stage s
    //
    //    Registered stages 1..3 are built with always_ff below.
    //    Stage 0 is combinatorially connected to the module input.
    // =========================================================================
    logic signed [7:0] a_stage [4][4]; // [row/col index][delay stage 0..3]
    logic signed [7:0] b_stage [4][4];
    logic              v_stage [4];    // valid delay stages 0..3

    // Stage 0: direct combinatorial tap
    genvar gi;
    generate
        for (gi = 0; gi < 4; gi++) begin : gen_stage0
            assign a_stage[gi][0] = a_in[gi];
            assign b_stage[gi][0] = b_in[gi];
        end
    endgenerate
    assign v_stage[0] = valid_in;

    // Stages 1..3: registered shift
    always_ff @(posedge clock) begin
        if (reset) begin
            for (int s = 1; s < 4; s++) begin
                for (int idx = 0; idx < 4; idx++) begin
                    a_stage[idx][s] <= 8'sd0;
                    b_stage[idx][s] <= 8'sd0;
                end
                v_stage[s] <= 1'b0;
            end
        end else begin
            for (int idx = 0; idx < 4; idx++) begin
                a_stage[idx][1] <= a_stage[idx][0];
                a_stage[idx][2] <= a_stage[idx][1];
                a_stage[idx][3] <= a_stage[idx][2];
                b_stage[idx][1] <= b_stage[idx][0];
                b_stage[idx][2] <= b_stage[idx][1];
                b_stage[idx][3] <= b_stage[idx][2];
            end
            v_stage[1] <= v_stage[0];
            v_stage[2] <= v_stage[1];
            v_stage[3] <= v_stage[2];
        end
    end

    // Skewed inputs: row/col index i uses delay stage i
    logic signed [7:0] a_skewed [4]; // a_skewed[i] = a_in[i] delayed i cycles
    logic signed [7:0] b_skewed [4]; // b_skewed[j] = b_in[j] delayed j cycles

    generate
        for (gi = 0; gi < 4; gi++) begin : gen_skewed
            assign a_skewed[gi] = a_stage[gi][gi];
            assign b_skewed[gi] = b_stage[gi][gi];
        end
    endgenerate

    // =========================================================================
    // 2. PE interconnect nets
    //    h_a[i][j]  : A wire entering column j of row i
    //                 j=0 → skewed input edge, j=1..4 → PE pass-through
    //    v_b[i][j]  : B wire entering row i of column j
    //                 i=0 → skewed input edge, i=1..4 → PE pass-through
    //    h_v[i][j]  : valid wire at the same position as h_a (travels with A)
    // =========================================================================
    logic signed [7:0] h_a [4][5]; // [row][col entry point 0..4]
    logic signed [7:0] v_b [5][4]; // [row entry point 0..4][col]
    logic              h_v [4][5]; // [row][col entry point 0..4]

    // Connect skewed inputs to the array left/top edges
    generate
        for (gi = 0; gi < 4; gi++) begin : gen_edge_connect
            assign h_a[gi][0] = a_skewed[gi]; // left edge of each row
            assign v_b[0][gi] = b_skewed[gi]; // top edge of each column
            assign h_v[gi][0] = v_stage[gi];  // valid enters with A, skewed per row
        end
    endgenerate

    // =========================================================================
    // 3. Instantiate 4×4 PE grid
    // =========================================================================
    genvar grow, gcol;
    generate
        for (grow = 0; grow < 4; grow++) begin : gen_row
            for (gcol = 0; gcol < 4; gcol++) begin : gen_col
                pe u_pe (
                    .clock     (clock),
                    .reset     (reset),
                    .clear     (clear),
                    .a_in      (h_a[grow][gcol]),
                    .b_in      (v_b[grow][gcol]),
                    .valid_in  (h_v[grow][gcol]),
                    .a_out     (h_a[grow][gcol+1]),
                    .b_out     (v_b[grow+1][gcol]),
                    .valid_out (h_v[grow][gcol+1]),
                    .acc_out   (c_out[grow][gcol])
                );
            end
        end
    endgenerate

    // valid_out: the valid flag that exits the right side of the bottom row
    // PE[3][3].valid_out = h_v[3][4] — indicates data has fully drained
    assign valid_out = h_v[3][4];

endmodule
