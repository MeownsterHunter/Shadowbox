block.u 0 L0 [] -> {
 r0  = ipa.constant a[0x80] rZ // delay=6 wr:0
 r1  = prmt r0 [0x4440] rZ     // delay=6 wt=000001
 r1  = shl.w r1 0x2            // delay=1
 r0  = shr.w.u32 r0 0x6        // delay=6
 r0  = lop2.or r1 r0           // delay=6
 r0  = lop2.xor r0 0x5a        // delay=6
 r1  = prmt r0 [0x4440] rZ     // delay=4
 r0  = popc r0                 // delay=6 wr:0
 r0  = i2f.f32.i32.re r0       // delay=4 wt=000001 wr:0
 r1  = i2f.f32.u32.re r1       // delay=2 wr:1
 r0  = fmul.ftz r0 0x3d000000  // delay=4 wt=000001
 r1  = fmul.ftz r1 0x3b808081  // delay=2 wt=000010
 r23 = mov r0                  // delay=4
 r0  = mov r1                  // delay=2
 r3  = mov 0x3f800000          // delay=1
 r2  = mov rZ                  // delay=1
 r1  = mov r23                 // delay=1
       exit                    // delay=13
       nop                     // delay=2
} -> []
