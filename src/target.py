class Target:
    class Arch:
        aarch64 = 'aarch64'  # ARM 64-bit architecture
        aarch64_32 = 'aarch64_32'  # ARM 64-bit with 32-bit compatibility
        aarch64_be = 'aarch64_be'  # Big-endian variant of ARM 64-bit
        amdgcn = 'amdgcn'  # AMD GCN (Graphics Core Next) architecture
        r600 = 'r600'  # AMD's Radeon R600 architecture
        arm = 'arm'  # ARM 32-bit architecture
        arm64 = 'arm64'  # ARM 64-bit architecture (another variant of aarch64)
        arm64_32 = 'arm64_32'  # ARM 64-bit architecture with 32-bit compatibility
        armeb = 'armeb'  # ARM architecture with big-endian byte order
        thumb = 'thumb'  # ARM Thumb instruction set (16-bit compressed instructions)
        thumbeb = 'thumbeb'  # ARM Thumb instruction set with big-endian byte order
        avr = 'avr'  # AVR architecture (used in embedded systems like Arduino)
        bpf = 'bpf'  # Berkeley Packet Filter (used for network packet filtering)
        bpfeb = 'bpfeb'  # Big-endian BPF architecture
        bpfel = 'bpfel'  # Little-endian BPF architecture
        hexagon = 'hexagon'  # Qualcomm Hexagon DSP architecture
        lanai = 'lanai'  # Lattice Semiconductor's Lanai architecture
        loongarch32 = 'loongarch32'  # Loongson 32-bit architecture
        loongarch64 = 'loongarch64'  # Loongson 64-bit architecture
        m68k = 'm68k'  # Motorola 68000 architecture
        mips = 'mips'  # MIPS 32-bit architecture
        mips64 = 'mips64'  # MIPS 64-bit architecture
        mips64el = 'mips64el'  # MIPS 64-bit little-endian architecture
        mipsel = 'mipsel'  # MIPS 32-bit little-endian architecture
        msp430 = 'msp430'  # Texas Instruments MSP430 architecture (low-power microcontrollers)
        nvptx = 'nvptx'  # NVIDIA PTX architecture (parallel threads for GPUs)
        nvptx64 = 'nvptx64'  # 64-bit variant of NVIDIA PTX
        ppc32 = 'ppc32'  # PowerPC 32-bit architecture
        ppc32le = 'ppc32le'  # PowerPC 32-bit little-endian architecture
        ppc64 = 'ppc64'  # PowerPC 64-bit architecture
        ppc64le = 'ppc64le'  # PowerPC 64-bit little-endian architecture
        riscv32 = 'riscv32'  # RISC-V 32-bit architecture
        riscv64 = 'riscv64'  # RISC-V 64-bit architecture
        sparc = 'sparc'  # SPARC architecture (Scalable Processor Architecture)
        sparcel = 'sparcel'  # SPARC little-endian variant
        sparcv9 = 'sparcv9'  # SPARC 64-bit architecture
        systemz = 'systemz'  # IBM System z architecture (mainframe systems)
        ve = 've'  # Vector Engine architecture (used in some supercomputers)
        wasm32 = 'wasm32'  # WebAssembly 32-bit architecture
        wasm64 = 'wasm64'  # WebAssembly 64-bit architecture
        x86 = 'x86'  # x86 32-bit architecture (Intel/AMD)
        x86_64 = 'x86_64'  # x86 64-bit architecture (also known as x64)
        xcore = 'xcore'  # XMOS XCore architecture (used in embedded systems)
        xtensa = 'xtensa'  # Tensilica Xtensa architecture (used in embedded and DSP applications)

    class Os:
        linux = 'linux'
        windows = 'windows'
        macos = 'macos'
        wasi = 'wasi'
        freebsd = 'freebsd'
        openbsd = 'openbsd'
        templeos = 'templeos'
        solaris = 'solaris'
        dragonfly = 'dragonfly'
        serenity = 'serenity'
        none = 'none'        # For bare-metal targets
        unknown = "unknown"  # Default value 
        
    class Abi:
        unknown = 'unknown'  # Default value for unknown ABI
        gnu = 'gnu'          # GNU ABI (used with Linux)
        gnueabi = 'gnueabi'  # GNU EABI (Embedded ABI)
        gnueabihf = 'gnueabihf'  # GNU EABI with hardware floating point
        msvc = 'msvc'        # Microsoft Visual C++ ABI
        android = 'android'  # Android ABI
        musl = 'musl'        # Musl libc ABI
        eabi = 'eabi'        # Embedded ABI
        eabihf = 'eabihf'    # Embedded ABI with hardware floating point
        cygnus = 'cygnus'    # Cygnus ABI (used with Cygwin)
        newlib = 'newlib'    # Newlib ABI
        mingw = 'mingw'      # MinGW ABI
        none = 'none'        # For bare-metal targets
        
    class Vendor:
        unknown = 'unknown'  # Default value for unknown vendor
        pc = 'pc'            # Personal Computer
        apple = 'apple'      # Apple Inc.
        ibm = 'ibm'          # IBM
        nvidia = 'nvidia'    # NVIDIA
        amd = 'amd'          # AMD
        arm = 'arm'          # ARM
        intel = 'intel'      # Intel
        microsoft = 'microsoft'  # Microsoft
        mips = 'mips'        # MIPS Technologies
        none = 'none'        # No specific vendor

    def __init__(self, arch: str, os: str = 'unknown', vendor: str = 'unknown', abi: str = 'unknown'):
        """
        Initialize a Target object with architecture, vendor, operating system, and optional ABI.
        Default values for vendor, os and abi are 'unknown'.
        """
        if arch not in vars(self.Arch).values():
            raise ValueError(f"Invalid architecture: {arch}")
        if os not in vars(self.Os).values():
            raise ValueError(f"Invalid OS: {os}")
        
        self.arch = arch
        self.vendor = vendor
        self.os = os
        self.abi = abi

    @property
    def triple(self) -> str:
        """
        Generate the LLVM triple based on the architecture, vendor, OS, and ABI.
        Format: <arch>-<vendor>-<os>-<abi>
        """
        return f"{self.arch}-{self.vendor}-{self.os}-{self.abi}"

    def get_llvm_triple(self) -> str:
        """
        Generate the LLVM triple based on the architecture, vendor, OS, and ABI.
        Format: <arch>-<vendor>-<os>-<abi>
        """
        return self.triple

    @staticmethod
    def from_string(target_str):
        """
        Parses a target string in one of the formats:
        - '<arch>-<vendor>-<os>-<abi>'
        - '<arch>-<vendor>-<os>'
        - '<arch>-<os>-<abi>'
        - '<arch>-<os>'
        and returns a Target instance.
        """
        parts = target_str.split('-')
        
        if len(parts) == 4:
            arch, vendor, os, abi = parts
        elif len(parts) == 3:
            # Could be either arch-vendor-os or arch-os-abi
            # Try to determine which by checking if the second part is a known vendor
            arch = parts[0]
            if parts[1] in vars(Target.Vendor).values():
                vendor, os = parts[1], parts[2]
                abi = 'unknown'
            else:
                os, abi = parts[1], parts[2]
                vendor = 'unknown'
        elif len(parts) == 2:
            arch, os = parts
            vendor = 'unknown'
            abi = 'unknown'
        else:
            raise ValueError(f"Invalid target string format: {target_str}")
        
        # Validate the architecture and OS
        if arch not in vars(Target.Arch).values():
            raise ValueError(f"Invalid architecture: {arch}")
        if os not in vars(Target.Os).values():
            raise ValueError(f"Invalid OS: {os}")
        
        # Return the constructed Target
        return Target(arch=arch, vendor=vendor, os=os, abi=abi)
    
    @staticmethod
    def list_targets():
        """
        Print all available target options.
        """
        print("Available architectures:")
        for name, value in vars(Target.Arch).items():
            if not name.startswith('__'):
                print(f"  {value}")
        
        print("\nAvailable operating systems:")
        for name, value in vars(Target.Os).items():
            if not name.startswith('__'):
                print(f"  {value}")
        
        print("\nAvailable vendors:")
        for name, value in vars(Target.Vendor).items():
            if not name.startswith('__'):
                print(f"  {value}")
        
        print("\nAvailable ABIs:")
        for name, value in vars(Target.Abi).items():
            if not name.startswith('__'):
                print(f"  {value}")
        
        print("\nExample target strings:")
        print("  riscv64-unknown-none-none     (RISC-V 64-bit bare-metal)")
        print("  x86_64-pc-linux-gnu          (x86-64 Linux)")
        print("  arm-none-none-eabi            (ARM bare-metal with EABI)")
        print("  aarch64-apple-macos-unknown   (ARM64 macOS)")
    
    def __str__(self):
        return self.get_llvm_triple()
    
    def __repr__(self):
        return str(self)