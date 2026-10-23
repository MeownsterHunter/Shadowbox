block.u 0 L0 [] -> {
 r1      = ipa.pass a[0x7c] rZ                              // delay=6 wr:0
 r1      = mufu.rcp.f32 r1                                  // delay=6 wt=000001 wr:0
 r2      = ipa.pass_mul_w a[0x80] r1                        // delay=1 wt=000001 wr:0
 r3      = ipa.pass_mul_w a[0x84] r1                        // delay=1 wr:1
 r0      = mov rZ                                           // delay=1
 r8      = mov rZ                                           // delay=1
 r9      = mov 0x3f800000                                   // delay=1
 r10     = mov 0xf1                                         // delay=5
 r4..8   = tld4.g.2d.nodep.rgba bindless r2..4 r0           // delay=1 wt=000011 wr:0
 r12..16 = tex.2d.ll.aoffi.nodep.rgba bindless r2..4 r8..11 // delay=6 wr:1
 r0      = fadd.ftz r12 r4                                  // delay=1 wt=000011
 r1      = fadd.ftz r13 r5                                  // delay=1
 r2      = fadd.ftz r14 r6                                  // delay=1
 r3      = fadd.ftz r15 r7                                  // delay=1
           exit                                             // delay=13
           nop                                              // delay=2
} -> []
