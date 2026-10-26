block.u 0 L0 [] -> {
 r0  = ipa.pass a[0x7c] rZ       // delay=6 wr:0
 r0  = mufu.rcp.f32 r0           // delay=6 wt=000001 wr:0
 r1  = ipa.pass_mul_w a[0x80] r0 // delay=1 wt=000001 wr:0
 r2  = ipa.pass_mul_w a[0x90] r0 // delay=6 wr:1
 r1  = fadd.ftz r1 r2            // delay=4 wt=000011
 r2  = ipa.pass_mul_w a[0x84] r0 // delay=1 wr:0
 r3  = ipa.pass_mul_w a[0x94] r0 // delay=6 wr:1
 r2  = fadd.ftz r2 r3            // delay=4 wt=000011
 r3  = ipa.pass_mul_w a[0x88] r0 // delay=1 wr:0
 r4  = ipa.pass_mul_w a[0x98] r0 // delay=6 wr:1
 r3  = fadd.ftz r3 r4            // delay=4 wt=000011
 r4  = ipa.pass_mul_w a[0x8c] r0 // delay=4 rd:0 wr:1
 r0  = ipa.pass_mul_w a[0x9c] r0 // delay=6 wt=000001 wr:0
 r0  = fadd.ftz r4 r0            // delay=6 wt=000011
 r23 = mov r0                    // delay=4
 r0  = mov r1                    // delay=4
 r1  = mov r2                    // delay=4
 r2  = mov r3                    // delay=4
 r3  = mov r23                   // delay=1
       exit                      // delay=13
       nop                       // delay=2
} -> []
