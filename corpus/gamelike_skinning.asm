block.u 0 L0 [] -> {
 r0..4   = ald a[0x90]             // delay=1 wr:0
 r4..8   = ald a[0xa0]             // delay=6 wr:1
 r5      = shl.w r5 0x6            // delay=6 wt=000010
 r8      = ldc.b32 c[0x1][r5]      // delay=1 wr:1
 r4      = shl.w r4 0x6            // delay=5
 r8      = fmul.ftz r8 r1          // delay=1 wt=000011
 r9      = ldc.b32 c[0x1][r4]      // delay=4 wr:0
 r6      = shl.w r6 0x6            // delay=2
 r8      = ffma.ftz r9 r0 r8       // delay=4 wt=000001
 r9      = ldc.b32 c[0x1][r6]      // delay=4 wr:0
 r7      = shl.w r7 0x6            // delay=2
 r8      = ffma.ftz r9 r2 r8       // delay=4 wt=000001
 r9      = ldc.b32 c[0x1][r7]      // delay=6 wr:0
 r8      = ffma.ftz r9 r3 r8       // delay=4 wt=000001
 r9      = ldc.b32 c[0x1][r5+0x10] // delay=6 wr:0
 r9      = fmul.ftz r9 r1          // delay=1 wt=000001
 r10     = ldc.b32 c[0x1][r4+0x10] // delay=6 wr:0
 r9      = ffma.ftz r10 r0 r9      // delay=4 wt=000001
 r10     = ldc.b32 c[0x1][r6+0x10] // delay=6 wr:0
 r9      = ffma.ftz r10 r2 r9      // delay=4 wt=000001
 r10     = ldc.b32 c[0x1][r7+0x10] // delay=6 wr:0
 r9      = ffma.ftz r10 r3 r9      // delay=4 wt=000001
 r10     = ldc.b32 c[0x1][r5+0x20] // delay=6 wr:0
 r10     = fmul.ftz r10 r1         // delay=1 wt=000001
 r11     = ldc.b32 c[0x1][r4+0x20] // delay=6 wr:0
 r10     = ffma.ftz r11 r0 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r6+0x20] // delay=6 wr:0
 r10     = ffma.ftz r11 r2 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r7+0x20] // delay=6 wr:0
 r10     = ffma.ftz r11 r3 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r5+0x30] // delay=6 wr:0
 r11     = fmul.ftz r11 r1         // delay=1 wt=000001
 r12     = ldc.b32 c[0x1][r4+0x30] // delay=6 wr:0
 r11     = ffma.ftz r12 r0 r11     // delay=4 wt=000001
 r12     = ldc.b32 c[0x1][r6+0x30] // delay=6 wr:0
 r11     = ffma.ftz r12 r2 r11     // delay=4 wt=000001
 r12     = ldc.b32 c[0x1][r7+0x30] // delay=6 wr:0
 r11     = ffma.ftz r12 r3 r11     // delay=4 wt=000001
 r12..15 = ald a[0x80]             // delay=6 wr:0
 r10     = ffma.ftz r10 r14 r11    // delay=6 wt=000001
 r9      = ffma.ftz r9 r13 r10     // delay=6
 r8      = ffma.ftz r8 r12 r9      // delay=4
 r9      = ldc.b32 c[0x1][r5+0x4]  // delay=6 wr:0
 r9      = fmul.ftz r9 r1          // delay=1 wt=000001
 r10     = ldc.b32 c[0x1][r4+0x4]  // delay=6 wr:0
 r9      = ffma.ftz r10 r0 r9      // delay=4 wt=000001
 r10     = ldc.b32 c[0x1][r6+0x4]  // delay=6 wr:0
 r9      = ffma.ftz r10 r2 r9      // delay=4 wt=000001
 r10     = ldc.b32 c[0x1][r7+0x4]  // delay=6 wr:0
 r9      = ffma.ftz r10 r3 r9      // delay=4 wt=000001
 r10     = ldc.b32 c[0x1][r5+0x14] // delay=6 wr:0
 r10     = fmul.ftz r10 r1         // delay=1 wt=000001
 r11     = ldc.b32 c[0x1][r4+0x14] // delay=6 wr:0
 r10     = ffma.ftz r11 r0 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r6+0x14] // delay=6 wr:0
 r10     = ffma.ftz r11 r2 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r7+0x14] // delay=6 wr:0
 r10     = ffma.ftz r11 r3 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r5+0x24] // delay=6 wr:0
 r11     = fmul.ftz r11 r1         // delay=1 wt=000001
 r15     = ldc.b32 c[0x1][r4+0x24] // delay=6 wr:0
 r11     = ffma.ftz r15 r0 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r6+0x24] // delay=6 wr:0
 r11     = ffma.ftz r15 r2 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r7+0x24] // delay=6 wr:0
 r11     = ffma.ftz r15 r3 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r5+0x34] // delay=6 wr:0
 r15     = fmul.ftz r15 r1         // delay=1 wt=000001
 r16     = ldc.b32 c[0x1][r4+0x34] // delay=6 wr:0
 r15     = ffma.ftz r16 r0 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r6+0x34] // delay=6 wr:0
 r15     = ffma.ftz r16 r2 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r7+0x34] // delay=6 wr:0
 r15     = ffma.ftz r16 r3 r15     // delay=6 wt=000001
 r11     = ffma.ftz r11 r14 r15    // delay=6
 r10     = ffma.ftz r10 r13 r11    // delay=6
 r9      = ffma.ftz r9 r12 r10     // delay=4
 r10     = ldc.b32 c[0x1][r5+0x8]  // delay=6 wr:0
 r10     = fmul.ftz r10 r1         // delay=1 wt=000001
 r11     = ldc.b32 c[0x1][r4+0x8]  // delay=6 wr:0
 r10     = ffma.ftz r11 r0 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r6+0x8]  // delay=6 wr:0
 r10     = ffma.ftz r11 r2 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r7+0x8]  // delay=6 wr:0
 r10     = ffma.ftz r11 r3 r10     // delay=4 wt=000001
 r11     = ldc.b32 c[0x1][r5+0x18] // delay=6 wr:0
 r11     = fmul.ftz r11 r1         // delay=1 wt=000001
 r15     = ldc.b32 c[0x1][r4+0x18] // delay=6 wr:0
 r11     = ffma.ftz r15 r0 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r6+0x18] // delay=6 wr:0
 r11     = ffma.ftz r15 r2 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r7+0x18] // delay=6 wr:0
 r11     = ffma.ftz r15 r3 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r5+0x28] // delay=6 wr:0
 r15     = fmul.ftz r15 r1         // delay=1 wt=000001
 r16     = ldc.b32 c[0x1][r4+0x28] // delay=6 wr:0
 r15     = ffma.ftz r16 r0 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r6+0x28] // delay=6 wr:0
 r15     = ffma.ftz r16 r2 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r7+0x28] // delay=6 wr:0
 r15     = ffma.ftz r16 r3 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r5+0x38] // delay=6 wr:0
 r16     = fmul.ftz r16 r1         // delay=1 wt=000001
 r17     = ldc.b32 c[0x1][r4+0x38] // delay=6 wr:0
 r16     = ffma.ftz r17 r0 r16     // delay=4 wt=000001
 r17     = ldc.b32 c[0x1][r6+0x38] // delay=6 wr:0
 r16     = ffma.ftz r17 r2 r16     // delay=4 wt=000001
 r17     = ldc.b32 c[0x1][r7+0x38] // delay=6 wr:0
 r16     = ffma.ftz r17 r3 r16     // delay=6 wt=000001
 r15     = ffma.ftz r15 r14 r16    // delay=6
 r11     = ffma.ftz r11 r13 r15    // delay=6
 r10     = ffma.ftz r10 r12 r11    // delay=4
 r11     = ldc.b32 c[0x1][r5+0xc]  // delay=6 wr:0
 r11     = fmul.ftz r11 r1         // delay=1 wt=000001
 r15     = ldc.b32 c[0x1][r4+0xc]  // delay=6 wr:0
 r11     = ffma.ftz r15 r0 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r6+0xc]  // delay=6 wr:0
 r11     = ffma.ftz r15 r2 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r7+0xc]  // delay=6 wr:0
 r11     = ffma.ftz r15 r3 r11     // delay=4 wt=000001
 r15     = ldc.b32 c[0x1][r5+0x1c] // delay=6 wr:0
 r15     = fmul.ftz r15 r1         // delay=1 wt=000001
 r16     = ldc.b32 c[0x1][r4+0x1c] // delay=6 wr:0
 r15     = ffma.ftz r16 r0 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r6+0x1c] // delay=6 wr:0
 r15     = ffma.ftz r16 r2 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r7+0x1c] // delay=6 wr:0
 r15     = ffma.ftz r16 r3 r15     // delay=4 wt=000001
 r16     = ldc.b32 c[0x1][r5+0x2c] // delay=4 rd:0 wr:1
 r5      = ldc.b32 c[0x1][r5+0x3c] // delay=2 wt=000001 wr:0
 r16     = fmul.ftz r16 r1         // delay=1 wt=000010
 r17     = ldc.b32 c[0x1][r4+0x2c] // delay=3 rd:1 wr:2
 r1      = fmul.ftz r5 r1          // delay=1 wt=000001
 r4      = ldc.b32 c[0x1][r4+0x3c] // delay=2 wt=000010 wr:0
 r16     = ffma.ftz r17 r0 r16     // delay=4 wt=000100
 r0      = ffma.ftz r4 r0 r1       // delay=3 wt=000001
 r17     = ldc.b32 c[0x1][r6+0x2c] // delay=1 wr:0
 r1      = ldc.b32 c[0x1][r6+0x3c] // delay=5 wr:1
 r16     = ffma.ftz r17 r2 r16     // delay=1 wt=000001
 r0      = ffma.ftz r1 r2 r0       // delay=3 wt=000010
 r17     = ldc.b32 c[0x1][r7+0x2c] // delay=1 wr:0
 r1      = ldc.b32 c[0x1][r7+0x3c] // delay=5 wr:1
 r16     = ffma.ftz r17 r3 r16     // delay=1 wt=000001
 r0      = ffma.ftz r1 r3 r0       // delay=6 wt=000010
 r0      = ffma.ftz r16 r14 r0     // delay=6
 r0      = ffma.ftz r15 r13 r0     // delay=6
 r11     = ffma.ftz r11 r12 r0     // delay=5
           ast a[0x80] r8..11      // delay=1 rd:0
           ast a[0x70] r8..12      // delay=2 rd:1
           exit                    // delay=13 wt=000011
           nop                     // delay=2
} -> []
