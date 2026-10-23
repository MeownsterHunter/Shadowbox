block.u 0 L0 [] -> {
 r2     = mov rZ                                 // delay=1
 r3     = mov rZ                                 // delay=6
 r4..7  = txq.rga bindless r2..4 dimension       // delay=4 rd:0 wr:1
 r2     = ipa.pass a[0x7c] rZ                    // delay=6 wt=000001 wr:0
 r2     = mufu.rcp.f32 r2                        // delay=1 wt=000001 wr:0
 p0     = isetp.eq.i32 r6 rZ                     // delay=12 wt=000010
 r3     = ipa.pass_mul_w a[0x80] r2              // delay=1 wt=000001 rd:0 wr:1
 r1     = sel p0 rZ r4                           // delay=6
 r1     = i2f.f32.i32.re r1                      // delay=5 wr:2
 r2     = ipa.pass_mul_w a[0x84] r2              // delay=1 wt=000001 wr:0
 r1     = fmul.ftz r3 r1                         // delay=6 wt=000110
 r6     = f2i.i32.f32.rz.ftz r1                  // delay=4 rd:1 wr:2
 r1     = sel p0 rZ r5                           // delay=6 wt=000010
 r1     = i2f.f32.i32.re r1                      // delay=6 wr:1
 r1     = fmul.ftz r2 r1                         // delay=6 wt=000011
 r7     = f2i.i32.f32.rz.ftz r1                  // delay=1 wr:0
 r0     = mov rZ                                 // delay=6
 r8..12 = tld.2d.lz.nodep.rgba bindless r6..8 r0 // delay=6 wt=000101 wr:0
 r3     = mov r11                                // delay=1 wt=000001
 r2     = mov r10                                // delay=1
 r1     = mov r9                                 // delay=1
 r0     = mov r8                                 // delay=1
          exit                                   // delay=13
          nop                                    // delay=2
} -> []
