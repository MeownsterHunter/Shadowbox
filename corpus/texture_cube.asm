block.u 0 L0 [] -> {
 r1    = ipa.pass a[0x7c] rZ                    // delay=6 wr:0
 r1    = mufu.rcp.f32 r1                        // delay=6 wt=000001 wr:0
 r2    = ipa.pass_mul_w a[0x80] r1              // delay=1 wt=000001 rd:0 wr:1
 r3    = ipa.pass_mul_w a[0x84] r1              // delay=5 rd:2 wr:3
 r1    = ipa.pass_mul_w a[0x88] r1              // delay=1 wt=000101 wr:0
 r4    = fmnmx.ftz |r2| |r3| pF                 // delay=6 wt=001010
 r4    = fmnmx.ftz r4 |r1| pF                   // delay=6 wt=000001
 r4    = mufu.rcp.f32 r4                        // delay=5 wr:0
 r0    = mov 0x100000                           // delay=1
 r8    = fmul.ftz r2 r4                         // delay=1 wt=000001
 r9    = fmul.ftz r3 r4                         // delay=1
 r10   = fmul.ftz r1 r4                         // delay=6
 r4..8 = tex.cube.nodep.rgba bindless r8..11 r0 // delay=6 wr:0
 r3    = mov r7                                 // delay=1 wt=000001
 r2    = mov r6                                 // delay=1
 r1    = mov r5                                 // delay=1
 r0    = mov r4                                 // delay=1
         exit                                   // delay=13
         nop                                    // delay=2
} -> []
