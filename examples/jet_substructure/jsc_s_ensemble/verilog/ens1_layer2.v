module ens1_layer2 (input [63:0] M0, output [63:0] M1);

wire [5:0] ens1_layer2_N0_wire = {M0[25], M0[24], M0[43], M0[42], M0[49], M0[48]};
ens1_layer2_N0 ens1_layer2_N0_inst (.M0(ens1_layer2_N0_wire), .M1(M1[1:0]));

wire [5:0] ens1_layer2_N1_wire = {M0[5], M0[4], M0[23], M0[22], M0[61], M0[60]};
ens1_layer2_N1 ens1_layer2_N1_inst (.M0(ens1_layer2_N1_wire), .M1(M1[3:2]));

wire [5:0] ens1_layer2_N2_wire = {M0[15], M0[14], M0[35], M0[34], M0[39], M0[38]};
ens1_layer2_N2 ens1_layer2_N2_inst (.M0(ens1_layer2_N2_wire), .M1(M1[5:4]));

wire [5:0] ens1_layer2_N3_wire = {M0[23], M0[22], M0[41], M0[40], M0[63], M0[62]};
ens1_layer2_N3 ens1_layer2_N3_inst (.M0(ens1_layer2_N3_wire), .M1(M1[7:6]));

wire [5:0] ens1_layer2_N4_wire = {M0[9], M0[8], M0[53], M0[52], M0[61], M0[60]};
ens1_layer2_N4 ens1_layer2_N4_inst (.M0(ens1_layer2_N4_wire), .M1(M1[9:8]));

wire [5:0] ens1_layer2_N5_wire = {M0[9], M0[8], M0[23], M0[22], M0[43], M0[42]};
ens1_layer2_N5 ens1_layer2_N5_inst (.M0(ens1_layer2_N5_wire), .M1(M1[11:10]));

wire [5:0] ens1_layer2_N6_wire = {M0[9], M0[8], M0[35], M0[34], M0[49], M0[48]};
ens1_layer2_N6 ens1_layer2_N6_inst (.M0(ens1_layer2_N6_wire), .M1(M1[13:12]));

wire [5:0] ens1_layer2_N7_wire = {M0[5], M0[4], M0[25], M0[24], M0[31], M0[30]};
ens1_layer2_N7 ens1_layer2_N7_inst (.M0(ens1_layer2_N7_wire), .M1(M1[15:14]));

wire [5:0] ens1_layer2_N8_wire = {M0[7], M0[6], M0[17], M0[16], M0[45], M0[44]};
ens1_layer2_N8 ens1_layer2_N8_inst (.M0(ens1_layer2_N8_wire), .M1(M1[17:16]));

wire [5:0] ens1_layer2_N9_wire = {M0[11], M0[10], M0[45], M0[44], M0[59], M0[58]};
ens1_layer2_N9 ens1_layer2_N9_inst (.M0(ens1_layer2_N9_wire), .M1(M1[19:18]));

wire [5:0] ens1_layer2_N10_wire = {M0[39], M0[38], M0[49], M0[48], M0[63], M0[62]};
ens1_layer2_N10 ens1_layer2_N10_inst (.M0(ens1_layer2_N10_wire), .M1(M1[21:20]));

wire [5:0] ens1_layer2_N11_wire = {M0[15], M0[14], M0[21], M0[20], M0[49], M0[48]};
ens1_layer2_N11 ens1_layer2_N11_inst (.M0(ens1_layer2_N11_wire), .M1(M1[23:22]));

wire [5:0] ens1_layer2_N12_wire = {M0[33], M0[32], M0[43], M0[42], M0[49], M0[48]};
ens1_layer2_N12 ens1_layer2_N12_inst (.M0(ens1_layer2_N12_wire), .M1(M1[25:24]));

wire [5:0] ens1_layer2_N13_wire = {M0[15], M0[14], M0[35], M0[34], M0[59], M0[58]};
ens1_layer2_N13 ens1_layer2_N13_inst (.M0(ens1_layer2_N13_wire), .M1(M1[27:26]));

wire [5:0] ens1_layer2_N14_wire = {M0[17], M0[16], M0[53], M0[52], M0[61], M0[60]};
ens1_layer2_N14 ens1_layer2_N14_inst (.M0(ens1_layer2_N14_wire), .M1(M1[29:28]));

wire [5:0] ens1_layer2_N15_wire = {M0[9], M0[8], M0[31], M0[30], M0[37], M0[36]};
ens1_layer2_N15 ens1_layer2_N15_inst (.M0(ens1_layer2_N15_wire), .M1(M1[31:30]));

wire [5:0] ens1_layer2_N16_wire = {M0[11], M0[10], M0[57], M0[56], M0[59], M0[58]};
ens1_layer2_N16 ens1_layer2_N16_inst (.M0(ens1_layer2_N16_wire), .M1(M1[33:32]));

wire [5:0] ens1_layer2_N17_wire = {M0[1], M0[0], M0[43], M0[42], M0[59], M0[58]};
ens1_layer2_N17 ens1_layer2_N17_inst (.M0(ens1_layer2_N17_wire), .M1(M1[35:34]));

wire [5:0] ens1_layer2_N18_wire = {M0[21], M0[20], M0[31], M0[30], M0[51], M0[50]};
ens1_layer2_N18 ens1_layer2_N18_inst (.M0(ens1_layer2_N18_wire), .M1(M1[37:36]));

wire [5:0] ens1_layer2_N19_wire = {M0[31], M0[30], M0[33], M0[32], M0[49], M0[48]};
ens1_layer2_N19 ens1_layer2_N19_inst (.M0(ens1_layer2_N19_wire), .M1(M1[39:38]));

wire [5:0] ens1_layer2_N20_wire = {M0[3], M0[2], M0[7], M0[6], M0[37], M0[36]};
ens1_layer2_N20 ens1_layer2_N20_inst (.M0(ens1_layer2_N20_wire), .M1(M1[41:40]));

wire [5:0] ens1_layer2_N21_wire = {M0[3], M0[2], M0[43], M0[42], M0[53], M0[52]};
ens1_layer2_N21 ens1_layer2_N21_inst (.M0(ens1_layer2_N21_wire), .M1(M1[43:42]));

wire [5:0] ens1_layer2_N22_wire = {M0[1], M0[0], M0[7], M0[6], M0[63], M0[62]};
ens1_layer2_N22 ens1_layer2_N22_inst (.M0(ens1_layer2_N22_wire), .M1(M1[45:44]));

wire [5:0] ens1_layer2_N23_wire = {M0[9], M0[8], M0[15], M0[14], M0[59], M0[58]};
ens1_layer2_N23 ens1_layer2_N23_inst (.M0(ens1_layer2_N23_wire), .M1(M1[47:46]));

wire [5:0] ens1_layer2_N24_wire = {M0[23], M0[22], M0[37], M0[36], M0[39], M0[38]};
ens1_layer2_N24 ens1_layer2_N24_inst (.M0(ens1_layer2_N24_wire), .M1(M1[49:48]));

wire [5:0] ens1_layer2_N25_wire = {M0[21], M0[20], M0[35], M0[34], M0[37], M0[36]};
ens1_layer2_N25 ens1_layer2_N25_inst (.M0(ens1_layer2_N25_wire), .M1(M1[51:50]));

wire [5:0] ens1_layer2_N26_wire = {M0[5], M0[4], M0[45], M0[44], M0[53], M0[52]};
ens1_layer2_N26 ens1_layer2_N26_inst (.M0(ens1_layer2_N26_wire), .M1(M1[53:52]));

wire [5:0] ens1_layer2_N27_wire = {M0[19], M0[18], M0[47], M0[46], M0[53], M0[52]};
ens1_layer2_N27 ens1_layer2_N27_inst (.M0(ens1_layer2_N27_wire), .M1(M1[55:54]));

wire [5:0] ens1_layer2_N28_wire = {M0[1], M0[0], M0[27], M0[26], M0[59], M0[58]};
ens1_layer2_N28 ens1_layer2_N28_inst (.M0(ens1_layer2_N28_wire), .M1(M1[57:56]));

wire [5:0] ens1_layer2_N29_wire = {M0[11], M0[10], M0[47], M0[46], M0[51], M0[50]};
ens1_layer2_N29 ens1_layer2_N29_inst (.M0(ens1_layer2_N29_wire), .M1(M1[59:58]));

wire [5:0] ens1_layer2_N30_wire = {M0[5], M0[4], M0[11], M0[10], M0[39], M0[38]};
ens1_layer2_N30 ens1_layer2_N30_inst (.M0(ens1_layer2_N30_wire), .M1(M1[61:60]));

wire [5:0] ens1_layer2_N31_wire = {M0[23], M0[22], M0[27], M0[26], M0[49], M0[48]};
ens1_layer2_N31 ens1_layer2_N31_inst (.M0(ens1_layer2_N31_wire), .M1(M1[63:62]));

endmodule