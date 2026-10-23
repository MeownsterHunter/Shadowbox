block.u 0 L0 [] -> {
 r1  = ipa.pass a[0x7c] rZ       // delay=6 wr:0
 r1  = mufu.rcp.f32 r1           // delay=6 wt=000001 wr:0
 r2  = ipa.pass_mul_w a[0x80] r1 // delay=1 wt=000001 rd:0 wr:1
 r3  = ipa.pass_mul_w a[0x84] r1 // delay=1 rd:2 wr:3
 r4  = ipa.pass_mul_w a[0x88] r1 // delay=1 rd:4 wr:5
 r0  = mov 0x3f000000            // delay=5
 r1  = ipa.pass_mul_w a[0x8c] r1 // delay=1 wt=010101 wr:0
 r0  = ffma.ftz r2 0x3fa00000 r0 // delay=4 wt=000010
 r2  = mov 0x3f000000            // delay=6
 r2  = ffma.ftz r3 0x3fa00000 r2 // delay=4 wt=001000
 r3  = mov 0x3f000000            // delay=6
 r3  = ffma.ftz r4 0x3fa00000 r3 // delay=4 wt=100000
 r4  = mov 0x3f000000            // delay=6
 r1  = ffma.ftz r1 0x3fa00000 r4 // delay=6 wt=000001
 r23 = mov r1                    // delay=4
 r1  = mov r2                    // delay=4
 r2  = mov r3                    // delay=4
 r3  = mov r23                   // delay=1
       exit                      // delay=13
       nop                       // delay=2
} -> []
