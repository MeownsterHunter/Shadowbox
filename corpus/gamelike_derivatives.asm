block.u 0 L0 [] -> {
 r0  = ipa.pass a[0x7c] rZ                      // delay=6 wr:0
 r0  = mufu.rcp.f32 r0                          // delay=6 wt=000001 wr:0
 r1  = ipa.pass_mul_w a[0x80] r0                // delay=5 wt=000001 rd:0 wr:1
 r0  = ipa.pass_mul_w a[0x84] r0                // delay=1 wt=000001 wr:0
 r2  = shfl.bfly r1 0x1 0x1c03                  // delay=4 wt=000010 wr:1
 r5  = shfl.bfly r1 0x2 0x1c03                  // delay=2 wr:2
 r2  = fswzadd.ftz r2 r1 [sub, subr, sub, subr] // delay=3 wt=000010
 r3  = shfl.bfly r0 0x1 0x1c03                  // delay=1 wt=000001 wr:0
 r1  = fswzadd.ftz r5 r1 [sub, sub, subr, subr] // delay=4 wt=000100
 r5  = shfl.bfly r0 0x2 0x1c03                  // delay=2 wr:1
 r3  = fswzadd.ftz r3 r0 [sub, subr, sub, subr] // delay=4 wt=000001
 r0  = fswzadd.ftz r5 r0 [sub, sub, subr, subr] // delay=5 wt=000010
 r4  = fmul.ftz r3 r3                           // delay=1
 r0  = fmul.ftz r0 r0                           // delay=5
 r4  = ffma.ftz r2 r2 r4                        // delay=1
 r0  = ffma.ftz r1 r1 r0                        // delay=5
 r4  = mufu.sqrt.f32 r4                         // delay=1 wr:0
 r0  = mufu.sqrt.f32 r0                         // delay=6 wr:1
 r0  = fadd.ftz r4 r0                           // delay=6 wt=000011
 r0  = fadd.ftz r1 r0                           // delay=6
 r23 = mov r0                                   // delay=4
 r0  = mov r2                                   // delay=2
 r1  = mov 0x3f800000                           // delay=2
 r2  = mov r23                                  // delay=4
 r23 = mov r1                                   // delay=4
 r1  = mov r3                                   // delay=4
 r3  = mov r23                                  // delay=1
       exit                                     // delay=13
       nop                                      // delay=2
} -> []
