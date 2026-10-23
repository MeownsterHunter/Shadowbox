block.u 0 L0 [] -> {
 r0 = ipa.constant a[0x80] rZ // delay=1 wr:0
 r1 = ipa.constant a[0x84] rZ // delay=1 wr:1
 r2 = ipa.constant a[0x88] rZ // delay=1 wr:2
 r3 = ipa.constant a[0x8c] rZ // delay=3 wr:3
 r0 = imnmx.i32 r0 0x7 pT     // delay=1 wt=000001
 r1 = imnmx.i32 r1 0x7 pT     // delay=1 wt=000010
 r2 = imnmx.i32 r2 0x7 pT     // delay=1 wt=000100
 r3 = imnmx.i32 r3 0x7 pT     // delay=3 wt=001000
 r0 = i2f.f32.i32.re r0       // delay=1 wr:0
 r1 = i2f.f32.i32.re r1       // delay=1 wr:1
 r2 = i2f.f32.i32.re r2       // delay=1 wr:2
 r3 = i2f.f32.i32.re r3       // delay=2 wr:3
      exit                    // delay=13 wt=001111
      nop                     // delay=2
} -> []
