block.u 0 L0 [] -> {
     r1    = ipa.pass a[0x7c] rZ                 // delay=6 wr:0
     r1    = mufu.rcp.f32 r1                     // delay=6 wt=000001 wr:0
     r2    = ipa.pass_mul_w a[0x80] r1           // delay=1 wt=000001 wr:0
     r3    = ipa.pass_mul_w a[0x84] r1           // delay=1 wr:1
     r0    = mov rZ                              // delay=6
     r4..8 = tex.2d.nodep.rgba bindless r2..4 r0 // delay=6 wt=000011 wr:0
     p0    = fsetp.lt.ftz r7 0x3f000000          // delay=13 wt=000001
@p0          kill                                // delay=13
             nop                                 // delay=2
     r3    = mov r7                              // delay=1
     r2    = mov r6                              // delay=1
     r1    = mov r5                              // delay=1
     r0    = mov r4                              // delay=1
             exit                                // delay=13
             nop                                 // delay=2
} -> []
