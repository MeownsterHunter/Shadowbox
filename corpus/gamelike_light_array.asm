block.u 0 L0 [] -> {
 r0  = ipa.pass a[0x7c] rZ         // delay=6 wr:0
 r0  = mufu.rcp.f32 r0             // delay=6 wt=000001 wr:0
 r5  = ipa.pass_mul_w a[0x88] r0   // delay=4 wt=000001 wr:0
 r3  = ipa.pass_mul_w a[0x84] r0   // delay=1 wr:1
 r1  = ipa.pass_mul_w a[0x80] r0   // delay=1 wr:2
 r6  = fadd.ftz r5 c[0x1][0x8]     // delay=5 wt=000001
 r4  = fadd.ftz r3 c[0x1][0x4]     // delay=1 wt=000010
 r7  = fmul.ftz r6 r6              // delay=5
 r2  = fadd.ftz r1 c[0x1][0x0]     // delay=1 wt=000100
 r7  = ffma.ftz r4 r4 r7           // delay=6
 r7  = ffma.ftz r2 r2 r7           // delay=6
 r7  = mufu.rsq.f32 r7             // delay=6 wr:0
 r2  = fmul.ftz r2 r7              // delay=1 wt=000001
 r4  = fmul.ftz r4 r7              // delay=1
 r6  = fmul.ftz r6 r7              // delay=4
 r7  = ipa.pass_mul_w a[0x98] r0   // delay=1 rd:0 wr:1
 r9  = ipa.pass_mul_w a[0x94] r0   // delay=4 rd:2 wr:3
 r0  = ipa.pass_mul_w a[0x90] r0   // delay=1 wt=000101 wr:0
 r8  = fmul.ftz r7 r7              // delay=6 wt=000010
 r8  = ffma.ftz r9 r9 r8           // delay=6 wt=001000
 r8  = ffma.ftz r0 r0 r8           // delay=6 wt=000001
 r8  = mufu.rsq.f32 r8             // delay=6 wr:0
 r7  = fmul.ftz r7 r8              // delay=6 wt=000001
 r6  = fmul.ftz r7 r6              // delay=1
 r9  = fmul.ftz r9 r8              // delay=1
 r0  = fmul.ftz r0 r8              // delay=4
 r8  = fadd.ftz r5 c[0x1][0x28]    // delay=1
 r4  = ffma.ftz r9 r4 r6           // delay=4
 r6  = fadd.ftz r3 c[0x1][0x24]    // delay=1
 r10 = fmul.ftz r8 r8              // delay=1
 r2  = ffma.ftz r0 r2 r4           // delay=4
 r4  = fadd.ftz r1 c[0x1][0x20]    // delay=1
 r10 = ffma.ftz r6 r6 r10          // delay=6
 r10 = ffma.ftz r4 r4 r10          // delay=6
 r10 = mufu.rsq.f32 r10            // delay=6 wr:0
 r8  = fmul.ftz r8 r10             // delay=3 wt=000001
 r4  = fmul.ftz r4 r10             // delay=1
 r6  = fmul.ftz r6 r10             // delay=1
 r11 = fadd.ftz r5 c[0x1][0x48]    // delay=1
 r8  = fmul.ftz r7 r8              // delay=4
 r10 = fadd.ftz r3 c[0x1][0x44]    // delay=1
 r12 = fmul.ftz r11 r11            // delay=1
 r6  = ffma.ftz r9 r6 r8           // delay=4
 r8  = fadd.ftz r1 c[0x1][0x40]    // delay=1
 r12 = ffma.ftz r10 r10 r12        // delay=6
 r12 = ffma.ftz r8 r8 r12          // delay=6
 r12 = mufu.rsq.f32 r12            // delay=6 wr:0
 r11 = fmul.ftz r11 r12            // delay=5 wt=000001
 r10 = fmul.ftz r10 r12            // delay=1
 r11 = fmul.ftz r7 r11             // delay=5
 r8  = fmul.ftz r8 r12             // delay=1
 r10 = ffma.ftz r9 r10 r11         // delay=4
 r5  = fadd.ftz r5 c[0x1][0x68]    // delay=2
 r8  = ffma.ftz r0 r8 r10          // delay=3
 r3  = fadd.ftz r3 c[0x1][0x64]    // delay=1
 r10 = fmul.ftz r5 r5              // delay=5
 r1  = fadd.ftz r1 c[0x1][0x60]    // delay=1
 r10 = ffma.ftz r3 r3 r10          // delay=6
 r10 = ffma.ftz r1 r1 r10          // delay=6
 r10 = mufu.rsq.f32 r10            // delay=6 wr:0
 r5  = fmul.ftz r5 r10             // delay=5 wt=000001
 r3  = fmul.ftz r3 r10             // delay=1
 r5  = fmul.ftz r7 r5              // delay=3
 r4  = ffma.ftz r0 r4 r6           // delay=2
 r1  = fmul.ftz r1 r10             // delay=1
 r3  = ffma.ftz r9 r3 r5           // delay=3
 r4  = fmnmx.ftz r4 rZ pF          // delay=3
 r0  = ffma.ftz r0 r1 r3           // delay=2
 r2  = fmnmx.ftz r2 rZ pF          // delay=1
 r6  = fmul.ftz r4 c[0x1][0x30]    // delay=1
 r3  = fmul.ftz r4 c[0x1][0x34]    // delay=4
 r4  = fmul.ftz r4 c[0x1][0x38]    // delay=1
 r6  = ffma.ftz r2 c[0x1][0x10] r6 // delay=1
 r3  = ffma.ftz r2 c[0x1][0x14] r3 // delay=1
 r8  = fmnmx.ftz r8 rZ pF          // delay=3
 r2  = ffma.ftz r2 c[0x1][0x18] r4 // delay=3
 r6  = ffma.ftz r8 c[0x1][0x50] r6 // delay=1
 r0  = fmnmx.ftz r0 rZ pF          // delay=1
 r3  = ffma.ftz r8 c[0x1][0x54] r3 // delay=1
 r2  = ffma.ftz r8 c[0x1][0x58] r2 // delay=4
 r1  = ffma.ftz r0 c[0x1][0x70] r6 // delay=1
 r3  = ffma.ftz r0 c[0x1][0x74] r3 // delay=1
 r2  = ffma.ftz r0 c[0x1][0x78] r2 // delay=4
 r0  = mov 0x3f800000              // delay=6
 r23 = mov r0                      // delay=4
 r0  = mov r1                      // delay=4
 r1  = mov r3                      // delay=4
 r3  = mov r23                     // delay=1
       exit                        // delay=13
       nop                         // delay=2
} -> []
