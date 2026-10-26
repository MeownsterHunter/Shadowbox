block.u 0 L0 [] -> {
 r4..7   = ald a[0x80]                   // delay=1 wr:0
 r8..11  = ald a[0x90]                   // delay=6 wr:1
 r3      = fmul.ftz r10 c[0x1][0x20]     // delay=6 wt=000010
 r3      = ffma.ftz r9 c[0x1][0x10] r3   // delay=6
 r16     = ffma.ftz r8 c[0x1][0x0] r3    // delay=4
 r3      = fmul.ftz r10 c[0x1][0x24]     // delay=6
 r3      = ffma.ftz r9 c[0x1][0x14] r3   // delay=6
 r17     = ffma.ftz r8 c[0x1][0x4] r3    // delay=4
 r3      = fmul.ftz r10 c[0x1][0x28]     // delay=6
 r3      = ffma.ftz r9 c[0x1][0x18] r3   // delay=5
 r9      = mov c[0x1][0xb0]              // delay=1
 r18     = ffma.ftz r8 c[0x1][0x8] r3    // delay=4
 r8      = mov c[0x1][0xa0]              // delay=1
 r9      = fmul.ftz r9 c[0x1][0x4c]      // delay=5
 r7      = mov c[0x1][0x90]              // delay=1
 r8      = ffma.ftz r8 c[0x1][0x48] r9   // delay=3
 r10     = mov c[0x1][0xb0]              // delay=2
 r3      = mov c[0x1][0x80]              // delay=1
 r7      = ffma.ftz r7 c[0x1][0x44] r8   // delay=2
 r9      = mov c[0x1][0xa0]              // delay=1
 r10     = fmul.ftz r10 c[0x1][0x5c]     // delay=3
 r3      = ffma.ftz r3 c[0x1][0x40] r7   // delay=1
 r8      = mov c[0x1][0x90]              // delay=1
 r11     = mov c[0x1][0xb0]              // delay=1
 r9      = ffma.ftz r9 c[0x1][0x58] r10  // delay=3
 r7      = mov c[0x1][0x80]              // delay=1
 r10     = mov c[0x1][0xa0]              // delay=1
 r11     = fmul.ftz r11 c[0x1][0x6c]     // delay=1
 r8      = ffma.ftz r8 c[0x1][0x54] r9   // delay=4
 r9      = mov c[0x1][0x90]              // delay=1
 r10     = ffma.ftz r10 c[0x1][0x68] r11 // delay=1
 r7      = ffma.ftz r7 c[0x1][0x50] r8   // delay=4
 r8      = mov c[0x1][0x80]              // delay=1
 r9      = ffma.ftz r9 c[0x1][0x64] r10  // delay=1
 r0      = mov c[0x1][0x30]              // delay=3
 r1      = mov c[0x1][0x34]              // delay=1
 r2      = mov c[0x1][0x38]              // delay=1
 r8      = ffma.ftz r8 c[0x1][0x60] r9   // delay=1
 r0      = ffma.ftz r6 c[0x1][0x20] r0   // delay=3 wt=000001
 r9      = mov c[0x1][0x3c]              // delay=1
 r1      = ffma.ftz r6 c[0x1][0x24] r1   // delay=1
 r2      = ffma.ftz r6 c[0x1][0x28] r2   // delay=1
 r0      = ffma.ftz r5 c[0x1][0x10] r0   // delay=2
 r10     = mov c[0x1][0xb0]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x2c] r9   // delay=1
 r1      = ffma.ftz r5 c[0x1][0x14] r1   // delay=1
 r2      = ffma.ftz r5 c[0x1][0x18] r2   // delay=1
 r0      = ffma.ftz r4 c[0x1][0x0] r0    // delay=1
 r9      = mov c[0x1][0xa0]              // delay=1
 r10     = fmul.ftz r10 c[0x1][0x7c]     // delay=1
 r5      = ffma.ftz r5 c[0x1][0x1c] r6   // delay=1
 r1      = ffma.ftz r4 c[0x1][0x4] r1    // delay=1
 r2      = ffma.ftz r4 c[0x1][0x8] r2    // delay=2
 r6      = mov c[0x1][0x90]              // delay=1
 r9      = ffma.ftz r9 c[0x1][0x78] r10  // delay=1
 r4      = ffma.ftz r4 c[0x1][0xc] r5    // delay=4
 r5      = mov c[0x1][0x80]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x74] r9   // delay=6
 r5      = ffma.ftz r5 c[0x1][0x70] r6   // delay=6
 r5      = fmul.ftz r5 r4                // delay=6
 r5      = ffma.ftz r8 r2 r5             // delay=6
 r5      = ffma.ftz r7 r1 r5             // delay=4
 r7      = mov c[0x1][0xb4]              // delay=4
 r8      = ffma.ftz r3 r0 r5             // delay=1
 r6      = mov c[0x1][0xa4]              // delay=1
 r7      = fmul.ftz r7 c[0x1][0x4c]      // delay=4
 r5      = mov c[0x1][0x94]              // delay=1
 r9      = mov c[0x1][0xb4]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x48] r7   // delay=3
 r3      = mov c[0x1][0x84]              // delay=1
 r7      = mov c[0x1][0xa4]              // delay=1
 r9      = fmul.ftz r9 c[0x1][0x5c]      // delay=1
 r5      = ffma.ftz r5 c[0x1][0x44] r6   // delay=4
 r6      = mov c[0x1][0x94]              // delay=1
 r7      = ffma.ftz r7 c[0x1][0x58] r9   // delay=1
 r3      = ffma.ftz r3 c[0x1][0x40] r5   // delay=2
 r10     = mov c[0x1][0xb4]              // delay=2
 r5      = mov c[0x1][0x84]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x54] r7   // delay=2
 r9      = mov c[0x1][0xa4]              // delay=1
 r10     = fmul.ftz r10 c[0x1][0x6c]     // delay=3
 r5      = ffma.ftz r5 c[0x1][0x50] r6   // delay=1
 r7      = mov c[0x1][0x94]              // delay=1
 r11     = mov c[0x1][0xb4]              // delay=1
 r9      = ffma.ftz r9 c[0x1][0x68] r10  // delay=3
 r6      = mov c[0x1][0x84]              // delay=1
 r10     = mov c[0x1][0xa4]              // delay=1
 r11     = fmul.ftz r11 c[0x1][0x7c]     // delay=1
 r7      = ffma.ftz r7 c[0x1][0x64] r9   // delay=4
 r9      = mov c[0x1][0x94]              // delay=1
 r10     = ffma.ftz r10 c[0x1][0x78] r11 // delay=1
 r6      = ffma.ftz r6 c[0x1][0x60] r7   // delay=4
 r7      = mov c[0x1][0x84]              // delay=1
 r9      = ffma.ftz r9 c[0x1][0x74] r10  // delay=6
 r7      = ffma.ftz r7 c[0x1][0x70] r9   // delay=6
 r7      = fmul.ftz r7 r4                // delay=6
 r6      = ffma.ftz r6 r2 r7             // delay=6
 r5      = ffma.ftz r5 r1 r6             // delay=2
 r7      = mov c[0x1][0xb8]              // delay=3
 r12..14 = ald a[0xa0]                   // delay=1 wr:0
 r9      = ffma.ftz r3 r0 r5             // delay=1
 r6      = mov c[0x1][0xa8]              // delay=1
 r7      = fmul.ftz r7 c[0x1][0x4c]      // delay=4
 r5      = mov c[0x1][0x98]              // delay=1
 r10     = mov c[0x1][0xb8]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x48] r7   // delay=3
 r3      = mov c[0x1][0x88]              // delay=1
 r7      = mov c[0x1][0xa8]              // delay=1
 r10     = fmul.ftz r10 c[0x1][0x5c]     // delay=1
 r5      = ffma.ftz r5 c[0x1][0x44] r6   // delay=4
 r6      = mov c[0x1][0x98]              // delay=1
 r7      = ffma.ftz r7 c[0x1][0x58] r10  // delay=1
 r3      = ffma.ftz r3 c[0x1][0x40] r5   // delay=3
 r11     = mov c[0x1][0xb8]              // delay=1
 r5      = mov c[0x1][0x88]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x54] r7   // delay=1
           ast a[0x80] r0..3             // delay=1 rd:1
           ast a[0x90] r16..19           // delay=1 rd:2
 r10     = mov c[0x1][0xa8]              // delay=1
 r11     = fmul.ftz r11 c[0x1][0x6c]     // delay=1
           ast a[0xa0] r12..14           // delay=2 wt=000001 rd:0
 r5      = ffma.ftz r5 c[0x1][0x50] r6   // delay=1
 r7      = mov c[0x1][0x98]              // delay=1
 r12     = mov c[0x1][0xb8]              // delay=1 wt=000001
 r10     = ffma.ftz r10 c[0x1][0x68] r11 // delay=3
 r6      = mov c[0x1][0x88]              // delay=1
 r11     = mov c[0x1][0xa8]              // delay=1
 r12     = fmul.ftz r12 c[0x1][0x7c]     // delay=1
 r7      = ffma.ftz r7 c[0x1][0x64] r10  // delay=4
 r10     = mov c[0x1][0x98]              // delay=1
 r11     = ffma.ftz r11 c[0x1][0x78] r12 // delay=1
 r6      = ffma.ftz r6 c[0x1][0x60] r7   // delay=4
 r7      = mov c[0x1][0x88]              // delay=1
 r10     = ffma.ftz r10 c[0x1][0x74] r11 // delay=6
 r7      = ffma.ftz r7 c[0x1][0x70] r10  // delay=6
 r7      = fmul.ftz r7 r4                // delay=6
 r6      = ffma.ftz r6 r2 r7             // delay=6
 r5      = ffma.ftz r5 r1 r6             // delay=2
 r7      = mov c[0x1][0xbc]              // delay=4
 r10     = ffma.ftz r3 r0 r5             // delay=1
 r6      = mov c[0x1][0xac]              // delay=1
 r7      = fmul.ftz r7 c[0x1][0x4c]      // delay=4
 r5      = mov c[0x1][0x9c]              // delay=1
 r11     = mov c[0x1][0xbc]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x48] r7   // delay=3
 r3      = mov c[0x1][0x8c]              // delay=1
 r7      = mov c[0x1][0xac]              // delay=1
 r11     = fmul.ftz r11 c[0x1][0x5c]     // delay=1
 r5      = ffma.ftz r5 c[0x1][0x44] r6   // delay=4
 r6      = mov c[0x1][0x9c]              // delay=1
 r7      = ffma.ftz r7 c[0x1][0x58] r11  // delay=1
 r3      = ffma.ftz r3 c[0x1][0x40] r5   // delay=2
 r12     = mov c[0x1][0xbc]              // delay=2
 r5      = mov c[0x1][0x8c]              // delay=1
 r6      = ffma.ftz r6 c[0x1][0x54] r7   // delay=2
 r11     = mov c[0x1][0xac]              // delay=1
 r12     = fmul.ftz r12 c[0x1][0x6c]     // delay=3
 r5      = ffma.ftz r5 c[0x1][0x50] r6   // delay=1
 r7      = mov c[0x1][0x9c]              // delay=1
 r13     = mov c[0x1][0xbc]              // delay=1
 r11     = ffma.ftz r11 c[0x1][0x68] r12 // delay=3
 r6      = mov c[0x1][0x8c]              // delay=1
 r12     = mov c[0x1][0xac]              // delay=1
 r13     = fmul.ftz r13 c[0x1][0x7c]     // delay=1
 r7      = ffma.ftz r7 c[0x1][0x64] r11  // delay=4
 r11     = mov c[0x1][0x9c]              // delay=1
 r12     = ffma.ftz r12 c[0x1][0x78] r13 // delay=1
 r6      = ffma.ftz r6 c[0x1][0x60] r7   // delay=4
 r7      = mov c[0x1][0x8c]              // delay=1
 r11     = ffma.ftz r11 c[0x1][0x74] r12 // delay=6
 r7      = ffma.ftz r7 c[0x1][0x70] r11  // delay=6
 r4      = fmul.ftz r7 r4                // delay=6
 r2      = ffma.ftz r6 r2 r4             // delay=6 wt=000010
 r1      = ffma.ftz r5 r1 r2             // delay=6
 r11     = ffma.ftz r3 r0 r1             // delay=6
           ast a[0x70] r8..12            // delay=2 rd:0
           exit                          // delay=13 wt=000101
           nop                           // delay=2
} -> []
