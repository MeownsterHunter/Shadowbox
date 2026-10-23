block.u 0 L0 [] -> {
 r2  = ipa.pass a[0x7c] rZ        // delay=6 wr:0
 r2  = mufu.rcp.f32 r2            // delay=6 wt=000001 wr:0
 r3  = ipa.pass_mul_w a[0x88] r2  // delay=1 wt=000001 wr:0
 r5  = ipa.pass_mul_w a[0x84] r2  // delay=1 wr:1
 r7  = ipa.pass_mul_w a[0xa8] r2  // delay=1 wr:2
 r6  = ipa.pass_mul_w a[0x80] r2  // delay=1 wr:3
 r9  = ipa.pass_mul_w a[0xa4] r2  // delay=1 wr:4
 r10 = ipa.pass_mul_w a[0xa0] r2  // delay=1 wr:5
 r4  = fmul.ftz r3 r3             // delay=6 wt=000001
 r4  = ffma.ftz r5 r5 r4          // delay=1 wt=000010
 r8  = fmul.ftz r7 r7             // delay=5 wt=000100
 r4  = ffma.ftz r6 r6 r4          // delay=1 wt=001000
 r8  = ffma.ftz r9 r9 r8          // delay=5 wt=010000
 r4  = mufu.rsq.f32 r4            // delay=1 wr:0
 r8  = ffma.ftz r10 r10 r8        // delay=6 wt=100000
 r8  = mufu.rsq.f32 r8            // delay=1 wr:1
 r5  = fmul.ftz r5 r4             // delay=1 wt=000001
 r3  = fmul.ftz r3 r4             // delay=4
 r4  = fmul.ftz r6 r4             // delay=4
 r6  = fmul.ftz r10 r8            // delay=4 wt=000010
 r10 = ipa.pass_mul_w a[0x98] r2  // delay=1 rd:0 wr:1
 r13 = ipa.pass_mul_w a[0x94] r2  // delay=1 rd:2 wr:3
 r7  = fmul.ftz r7 r8             // delay=4
 r2  = ipa.pass_mul_w a[0x90] r2  // delay=1 wt=000101 wr:0
 r9  = fmul.ftz r9 r8             // delay=1
 r11 = fmul.ftz r3 r7             // delay=6
 r12 = ffma.ftz r5 r9 r11         // delay=6
 r8  = ffma.ftz r4 r6 r12         // delay=4
 r12 = fmul.ftz r10 r10           // delay=6 wt=000010
 r12 = ffma.ftz r13 r13 r12       // delay=1 wt=001000
 r11 = ffma.ftz -r5 r9 -r11       // delay=5
 r12 = ffma.ftz r2 r2 r12         // delay=1 wt=000001
 r11 = ffma.ftz -r4 r6 r11        // delay=5
 r12 = mufu.rsq.f32 r12           // delay=1 wr:0
 r11 = fmul.ftz r11 0x40000000    // delay=6
 r5  = ffma.ftz -r5 r11 -r9       // delay=4
 r9  = fmul.ftz r10 r12           // delay=1 wt=000001
 r3  = ffma.ftz -r3 r11 -r7       // delay=1
 r4  = ffma.ftz -r4 r11 -r6       // delay=4
 r6  = fmul.ftz r13 r12           // delay=1
 r3  = fmul.ftz r3 r9             // delay=5
 r2  = fmul.ftz r2 r12            // delay=1
 r3  = ffma.ftz r5 r6 r3          // delay=6
 r2  = ffma.ftz r4 r2 r3          // delay=6
 r2  = fmnmx.ftz r2 rZ pF         // delay=6
 r2  = mufu.log2.f32 r2           // delay=6 wr:0
 r2  = fmul.ftz r2 0x41800000     // delay=6 wt=000001
 r2  = rro.exp2 r2                // delay=6 wr:0
 r2  = mufu.exp2.f32 r2           // delay=5 wt=000001 wr:0
 r8  = fmnmx.ftz r8 rZ pF         // delay=1
 r3  = fmul.ftz r2 c[0x1][0x10]   // delay=6 wt=000001
 r3  = ffma.ftz r8 c[0x1][0x0] r3 // delay=4
 r0  = mov c[0x1][0x0]            // delay=1
 r1  = mov 0x3e4ccccc             // delay=1
 r3  = fmul.ftz r3 0x3f4ccccd     // delay=1
 r4  = fmul.ftz r2 c[0x1][0x14]   // delay=5
 r0  = ffma.ftz r0 r1 r3          // delay=1
 r4  = ffma.ftz r8 c[0x1][0x4] r4 // delay=4
 r1  = mov c[0x1][0x4]            // delay=1
 r3  = mov 0x3e4ccccc             // delay=1
 r4  = fmul.ftz r4 0x3f4ccccd     // delay=1
 r2  = fmul.ftz r2 c[0x1][0x18]   // delay=5
 r1  = ffma.ftz r1 r3 r4          // delay=1
 r2  = ffma.ftz r8 c[0x1][0x8] r2 // delay=4
 r3  = mov c[0x1][0x8]            // delay=1
 r4  = mov 0x3e4ccccc             // delay=1
 r2  = fmul.ftz r2 0x3f4ccccd     // delay=6
 r2  = ffma.ftz r3 r4 r2          // delay=4
 r3  = mov c[0x1][0xc]            // delay=1
       exit                       // delay=13
       nop                        // delay=2
} -> []
