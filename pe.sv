// =============================================================================
// pe.sv  —  Processing Element for INT8 Systolic Array
// =============================================================================
//
// Output-stationary PE: accumulates acc += a_in * b_in each cycle that
// valid_in is asserted.  A data passes through horizontally (a_out feeds
// the PE to the right); B data passes through vertically (b_out feeds
// the PE below).  The valid flag travels with A so the controller can
// track when data has drained through each column.
//
// Ports
// -----
//   clock      : system clock (rising-edge triggered)
//   reset      : synchronous active-high reset — zeros all registers
//   clear      : synchronous accumulator clear (takes priority over accumulate)
//   a_in       : INT8 signed activation input from the left
//   b_in       : INT8 signed weight input from above
//   valid_in   : qualifier — accumulate only when high
//   a_out      : registered pass-through of a_in  (to right neighbour)
//   b_out      : registered pass-through of b_in  (to bottom neighbour)
//   valid_out  : registered pass-through of valid_in (travels with A)
//   acc_out    : INT32 signed accumulated result (combinatorial read-out)
//
// Timing
// ------
//   All outputs register on the rising edge.
//   acc_out is readable at any time; the matmul controller knows when it
//   is stable after K accumulation cycles + pipeline-drain latency.
//
// =============================================================================

module pe (
    input  logic               clock,
    input  logic               reset,
    input  logic               clear,

    // Data inputs
    input  logic signed [7:0]  a_in,
    input  logic signed [7:0]  b_in,
    input  logic               valid_in,

    // Pass-through outputs (feed neighbouring PEs)
    output logic signed [7:0]  a_out,
    output logic signed [7:0]  b_out,
    output logic               valid_out,

    // Accumulated result
    output logic signed [31:0] acc_out
);

    // -------------------------------------------------------------------------
    // Multiply-accumulate and pass-through registers
    // -------------------------------------------------------------------------
    always_ff @(posedge clock) begin
        if (reset) begin
            a_out     <= 8'sd0;
            b_out     <= 8'sd0;
            valid_out <= 1'b0;
            acc_out   <= 32'sd0;
        end else begin
            // Pass-through: A flows right, B flows down, valid travels with A
            a_out     <= a_in;
            b_out     <= b_in;
            valid_out <= valid_in;

            // Accumulator: clear takes priority, then accumulate on valid
            if (clear) begin
                acc_out <= 32'sd0;
            end else if (valid_in) begin
                acc_out <= acc_out + (32'(signed'(a_in)) * 32'(signed'(b_in)));
            end
        end
    end

endmodule
