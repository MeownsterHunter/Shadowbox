block.u 0 L0 [] -> {
 r0  = ipa.pass a[0x7c] rZ       // delay=6 wr:0
 r0  = mufu.rcp.f32 r0           // delay=6 wt=000001 wr:0
 r1  = ipa.pass_mul_w a[0x80] r0 // delay=1 wt=000001 rd:0 wr:1
 r2  = ipa.pass_mul_w a[0x84] r0 // delay=1 rd:2 wr:3
 r3  = ipa.pass_mul_w a[0x88] r0 // delay=1 rd:4 wr:5
 r8  = ipa.pass_mul_w a[0x98] r0 // delay=1 wt=000011 rd:0 wr:1
 r4  = ipa.pass_mul_w a[0x8c] r0 // delay=1 wt=001100 rd:2 wr:3
 r6  = ipa.pass_mul_w a[0x90] r0 // delay=4 wt=110000 rd:4 wr:5
 r0  = ipa.pass_mul_w a[0x94] r0 // delay=1 wt=010101 wr:0
 r9  = fmul.ftz r8 r8            // delay=6 wt=000010
 r9  = ffma.ftz r0 r0 r9         // delay=2 wt=000001
 r7  = fmul.ftz r6 0x3f000000    // delay=4 wt=100000
 r6  = ffma.ftz r6 r6 r9         // delay=6
 r6  = mufu.rsq.f32 r6           // delay=1 wr:0
 r5  = mov 0x3f000000            // delay=6
 r5  = ffma.ftz r7 r6 r5         // delay=4 wt=000001
 r7  = mov 0x3f000000            // delay=1
 r0  = fmul.ftz r0 0x3f000000    // delay=6
 r0  = ffma.ftz r0 r6 r7         // delay=6
 r23 = mov r0                    // delay=2
 r7  = mov 0x3f000000            // delay=1
 r9  = fmul.ftz r8 r6            // delay=1
 r0  = mov r1                    // delay=4
 r1  = mov r2                    // delay=2
 r6  = ffma.ftz r9 0x3f000000 r7 // delay=2
 r2  = mov r3                    // delay=2
 r7  = mov 0x3f800000            // delay=1
 r8  = mov r4                    // delay=1 wt=001000
 r3  = mov r4                    // delay=4
 r4  = mov r5                    // delay=2
 r11 = mov r7                    // delay=1
 r10 = mov rZ                    // delay=1
 r5  = mov r23                   // delay=1
       exit                      // delay=13
       nop                       // delay=2
} -> []
