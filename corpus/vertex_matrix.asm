block.u 0 L0 [] -> {
 r0..4 = ald a[0x80]                 // delay=6 wr:0
 r4    = fmul.ftz r3 c[0x1][0x30]    // delay=2 wt=000001
 r5    = fmul.ftz r3 c[0x1][0x34]    // delay=1
 r6    = fmul.ftz r3 c[0x1][0x38]    // delay=3
 r4    = ffma.ftz r2 c[0x1][0x20] r4 // delay=1
 r3    = fmul.ftz r3 c[0x1][0x3c]    // delay=1
 r5    = ffma.ftz r2 c[0x1][0x24] r5 // delay=1
 r6    = ffma.ftz r2 c[0x1][0x28] r6 // delay=3
 r4    = ffma.ftz r1 c[0x1][0x10] r4 // delay=1
 r2    = ffma.ftz r2 c[0x1][0x2c] r3 // delay=1
 r5    = ffma.ftz r1 c[0x1][0x14] r5 // delay=1
 r6    = ffma.ftz r1 c[0x1][0x18] r6 // delay=4
 r1    = ffma.ftz r1 c[0x1][0x1c] r2 // delay=3
 r4    = ffma.ftz r0 c[0x1][0x0] r4  // delay=1
 r5    = ffma.ftz r0 c[0x1][0x4] r5  // delay=1
 r6    = ffma.ftz r0 c[0x1][0x8] r6  // delay=1
 r7    = ffma.ftz r0 c[0x1][0xc] r1  // delay=6
         ast a[0x70] r4..8           // delay=2 rd:0
         exit                        // delay=13 wt=000001
         nop                         // delay=2
} -> []
