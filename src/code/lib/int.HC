#define NULL    0
#define TRUE    1
#define FALSE   0
#define ON      1
#define OFF     0
#define I8_MIN (-0x80)
#define I8_MAX 0x7F
#define U8_MIN 0
#define U8_MAX 0xFF
#define I16_MIN (-0x8000)
#define I16_MAX 0x7FFF
#define U16_MIN 0
#define U16_MAX 0xFFFF
#define I32_MIN (-0x80000000)
#define I32_MAX 0x7FFFFFFF
#define U32_MIN 0
#define U32_MAX 0xFFFFFFFF
#define I64_MIN (-0x8000000000000000)
#define I64_MAX 0x7FFFFFFFFFFFFFFF
#define U64_MIN 0
#define U64_MAX 0xFFFFFFFFFFFFFFFF
#define INVALID_PTR     I64_MAX
#define STR_LEN 144

//(Int to F64 conversion is signed)
//Turn off 80-bit float constants with OPTf_NO_BUILTIN_CONST.
#define U64_F64_MAX (0x43F0000000000000(F64))
#define F64_MAX (0x7FEFFFFFFFFFFFFF(F64))
#define F64_MIN (0xFFEFFFFFFFFFFFFF(F64))
#define inf     (0x7FF0000000000000(F64))
#define inf     (0x7FF0000000000000(F64))
#define pi      (0x400921FB54442D18(F64))
#define pi      (0x400921FB54442D18(F64))
#define exp_1   (0x4005BF0A8B145769(F64)) //The number "e"
#define log2_10 (0x400A934F0979A371(F64))
#define log2_e  (0x3FF71547652B82FE(F64))
#define log10_2 (0x3FD34413509F79FF(F64))
#define loge_2  (0x3FE62E42FEFA39EF(F64))
#define sqrt2   (0x3FF6A09E667F3BCD(F64))
#define eps     (0x3CB0000000000000(F64))