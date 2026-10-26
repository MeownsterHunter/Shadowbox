block.u 0 L0 [] -> {
 r2      = ipa.pass a[0x7c] rZ                // delay=6 wr:0
 r2      = mufu.rcp.f32 r2                    // delay=6 wt=000001 wr:0
 r4      = ipa.pass_mul_w a[0x80] r2          // delay=1 wt=000001 wr:0
 r5      = ipa.pass_mul_w a[0x84] r2          // delay=1 wr:1
 r1      = mov rZ                             // delay=6
 r8..11  = tex.2d.nodep.rgb bindless r4..6 r1 // delay=4 wt=000011 rd:0 wr:1
 r1      = mov 0x100001                       // delay=6 wt=000001
 r12..15 = tex.2d.nodep.rgb bindless r4..6 r1 // delay=4 rd:0 wr:2
 r5      = ipa.pass_mul_w a[0x98] r2          // delay=1 wt=000001 rd:0 wr:3
 r7      = ipa.pass_mul_w a[0x94] r2          // delay=4 rd:4 wr:5
 r2      = ipa.pass_mul_w a[0x90] r2          // delay=1 wt=010001 wr:0
 r3      = mov 0xbf800000                     // delay=5
 r1      = mov 0xbf800000                     // delay=1
 r3      = ffma.ftz r14 0x40000000 r3         // delay=4 wt=000100
 r0      = mov 0xbf800000                     // delay=1
 r1      = ffma.ftz r13 0x40000000 r1         // delay=1
 r4      = fmul.ftz r3 r3                     // delay=5
 r0      = ffma.ftz r12 0x40000000 r0         // delay=1
 r4      = ffma.ftz r1 r1 r4                  // delay=1
 r6      = fmul.ftz r5 r5                     // delay=5 wt=001000
 r4      = ffma.ftz r0 r0 r4                  // delay=1
 r6      = ffma.ftz r7 r7 r6                  // delay=5 wt=100000
 r4      = mufu.rsq.f32 r4                    // delay=1 wr:2
 r6      = ffma.ftz r2 r2 r6                  // delay=6 wt=000001
 r6      = mufu.rsq.f32 r6                    // delay=1 wr:0
 r0      = fmul.ftz r0 r4                     // delay=1 wt=000100
 r1      = fmul.ftz r1 r4                     // delay=1
 r3      = fmul.ftz r3 r4                     // delay=4
 r4      = fmul.ftz r5 r6                     // delay=5 wt=000001
 r7      = fmul.ftz r7 r6                     // delay=1
 r3      = fmul.ftz r3 r4                     // delay=5
 r2      = fmul.ftz r2 r6                     // delay=1
 r1      = ffma.ftz r1 r7 r3                  // delay=6
 r0      = ffma.ftz r0 r2 r1                  // delay=6
 r0      = fmnmx.ftz r0 rZ pF                 // delay=6
 r1      = fmul.ftz r8 r0                     // delay=1 wt=000010
 r2      = fmul.ftz r9 r0                     // delay=4
 r0      = fmul.ftz r10 r0                    // delay=6
 r23     = mov r0                             // delay=4
 r0      = mov r1                             // delay=4
 r1      = mov r2                             // delay=3
 r3      = mov 0x3f800000                     // delay=1
 r2      = mov r23                            // delay=1
           exit                               // delay=13
           nop                                // delay=2
} -> []
