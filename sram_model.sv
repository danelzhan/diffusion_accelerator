// =============================================================================
// sram_model.sv  —  Parameterised Synchronous Single-Port SRAM Model
// =============================================================================
//
// Behavioural SRAM for simulation and FPGA BRAM inference.
// Supports independent write and read operations in the same cycle
// (write-first behaviour: a simultaneous read/write to the same address
// returns the NEW data on the next cycle).
//
// Parameters
// ----------
//   DEPTH  : number of addressable words  (default 256)
//   DWIDTH : data width in bits            (default 8 — INT8 for A/B SRAMs)
//            Set to 32 for the INT32 output (C) SRAM.
//
// Derived
// -------
//   ALEN   : address width = $clog2(DEPTH), auto-computed
//
// Ports
// -----
//   clock          : rising-edge triggered
//   reset          : synchronous active-high — zeros rdata only (mem retained)
//
//   Write channel  (combinatorial address + data, registered on clock edge)
//     we            : write enable
//     waddr [ALEN]  : write address
//     wdata [DWIDTH]: data to write
//
//   Read channel  (registered — 1-cycle read latency)
//     re            : read enable (gate; rdata holds last value when low)
//     raddr [ALEN]  : read address presented this cycle
//     rdata [DWIDTH]: data available the NEXT cycle
//
// Timing example (DWIDTH=8)
// -------------------------
//   Cycle 0: re=1, raddr=5  →  Cycle 1: rdata = mem[5]
//   Cycle 1: re=1, raddr=6  →  Cycle 2: rdata = mem[6]
//   Cycle 2: re=0           →  Cycle 3: rdata = mem[6]  (held)
//
// Usage in this project
// ---------------------
//   Instantiate one INT8  SRAM for the A matrix (activations).
//   Instantiate one INT8  SRAM for the B matrix (weights).
//   Instantiate one INT32 SRAM for the C matrix (output accumulation).
//   The testbench pre-loads A and B via the write port before asserting start.
//   The controller reads A/B and writes C via the respective read/write ports.
//
// =============================================================================

module sram_model #(
    parameter  int DEPTH  = 256,
    parameter  int DWIDTH = 8,
    localparam int ALEN   = $clog2(DEPTH)
) (
    input  logic              clock,
    input  logic              reset,

    // ── Write port ──────────────────────────────────────────────────────────
    input  logic              we,
    input  logic [ALEN-1:0]   waddr,
    input  logic [DWIDTH-1:0] wdata,

    // ── Read port (1-cycle latency) ──────────────────────────────────────────
    input  logic              re,
    input  logic [ALEN-1:0]   raddr,
    output logic [DWIDTH-1:0] rdata
);

    // -------------------------------------------------------------------------
    // Memory array
    // -------------------------------------------------------------------------
    logic [DWIDTH-1:0] mem [0:DEPTH-1];

    // -------------------------------------------------------------------------
    // Write — synchronous, no reset (memory content preserved across reset)
    // -------------------------------------------------------------------------
    always_ff @(posedge clock) begin
        if (we)
            mem[waddr] <= wdata;
    end

    // -------------------------------------------------------------------------
    // Read — synchronous, 1-cycle latency
    // rdata holds its value when re is de-asserted (registered output).
    // -------------------------------------------------------------------------
    always_ff @(posedge clock) begin
        if (reset) begin
            rdata <= '0;
        end else if (re) begin
            rdata <= mem[raddr];
        end
    end

endmodule
