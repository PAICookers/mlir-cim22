# BF16 Supplier Snapshots

The four fixed cases contain 32 tracked supplier reference files. Weight files reuse the existing `../<case_id>/cim_w.txt` snapshots. Lit reads the tracked files directly; line endings do not affect parsed values.

Case directories:

- [cim22_tb_0p9_macro1](cim22_tb_0p9_macro1/), [cim22_tb_0p9_macro2](cim22_tb_0p9_macro2/).
- [test_data_1_17_macro1](test_data_1_17_macro1/), [test_data_1_17_macro2](test_data_1_17_macro2/).

The retained files contain these independent stages:

- `cache*_in_bf16.txt`: eight addressed rows of 64 raw BF16 input elements.
- `cache*_in_bf16_processed.txt`: fixed prealigned signed INT8 values in 16-bit slots; `cache*_expmax.txt`: the corresponding maximum input exponents.
- `cache*_in_bf16_splited.txt`: the supplier's 16 input-body words per row.
- `cim*_w_exp.txt`: 16 exponent bytes in lane order; `tb_program_modifier_w*_exp.txt`: two fixed little-endian body words.
- `output_*.txt`: eight rows of 16 signed INT21 intermediate tokens.
- `output_bf16_*.txt`: final BF16 bits in the low 16 bits of each 21-bit token. Output text lists lane 15 first; runner results list lane 0 first.

The current replay checks input Cache words, Weight_EXP packing and BF16 response layout. Three mutations verify that each layout comparison rejects discrepancies. Historical prealignment and arithmetic outputs remain unchanged as evidence; they are not numerical expectations for the selected candidate runner. Candidate arithmetic is tested separately in `BF16SupportTest.cpp` and `CIMSoftwareRunnerTest.cpp`.

Response words are constructed by concatenating the fixed output tokens and 48 padding bits. This tests the adopted provisional layout, not an actual RTL response.

Evidence level: **historical-layout-match**, not RTL/board-verified. These snapshots contain already-prealigned integer weights and separately generated exponents; they cannot establish raw BF16 weight preprocessing or IEEE equivalence.
