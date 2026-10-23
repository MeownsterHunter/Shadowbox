block.u 0 L0 [] -> {
 r0..4 = ald a[0x80]       // delay=6 wr:0
         ast a[0x70] r0..4 // delay=2 wt=000001 rd:0
         exit              // delay=13 wt=000001
         nop               // delay=2
} -> []
