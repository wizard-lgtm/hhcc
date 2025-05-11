U8 main(){
// --- Integer to Integer ---
U8 a = 5;
U16 b = a;         // zext
U32 c = b;         // zext
U64 d = c;         // zext

U64 e = 1000;
U32 f = e;         // trunc
U16 g = f;         // trunc
U8 h = g;          // trunc

I8 sa = -5;
I16 sb = sa;       // sext
I32 sc = sb;       // sext
I64 sd = sc;       // sext

// --- Integer to Float ---
U8 u8_val = 10;
F64 f1 = u8_val;   // uitofp

I32 i32_val = -123;
F64 f2 = i32_val;  // sitofp



// --- Pointer to Integer and back ---
U8* p1 = &a;
U64 ip = p1;        // ptrtoint
U8* p2 = ip;        // inttoptr

// --- Pointer to Pointer ---
I32* ptr_a = 0;
U64* ptr_b = ptr_a; // bitcast

// --- Bool to Integer ---
Bool flag = true;
U8 flag_as_u8 = flag; // zext

// --- Integer to Bool ---
U16 num = 12;
Bool result = num;     // icmp != 0

// --- No-op (same type) ---
U64 unchanged = e; // same type pass-through

}
