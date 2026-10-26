block.u 0 L0 [] -> {
 r1     = ipa.pass a[0x7c] rZ                  // delay=6 wr:0
 r1     = mufu.rcp.f32 r1                      // delay=6 wt=000001 wr:0
 r2     = ipa.pass_mul_w a[0x88] r1            // delay=5 wt=000001 wr:0
 r5     = ipa.pass_mul_w a[0x80] r1            // delay=1 wr:1
 r2     = fadd.ftz r2 0x3f000000               // delay=5 wt=000001
 r6     = ipa.pass_mul_w a[0x84] r1            // delay=1 wr:0
 r2     = fmnmx.ftz r2 rZ pF                   // delay=6
 r2     = f2i.u32.f32.rz.ftz r2                // delay=5 wr:2
 r0     = mov 0x100000                         // delay=1
 r4     = imnmx.u32 r2 0xffff pT               // delay=6 wt=000100
 r8..12 = tex.a2d.nodep.rgba bindless r4..7 r0 // delay=6 wt=000011 wr:0
 r3     = mov r11                              // delay=1 wt=000001
 r2     = mov r10                              // delay=1
 r1     = mov r9                               // delay=1
 r0     = mov r8                               // delay=1
          exit                                 // delay=13
          nop                                  // delay=2
} -> []
