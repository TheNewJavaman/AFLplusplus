cmake_minimum_required(VERSION 3.20)
project(CoquiPassPlugin CXX)

find_package(LLVM 18.1 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COQUI_PASS_SOURCES
    CoquiPassPlugin.cpp
    RejectInlineAsm.cpp
    RejectLibc.cpp
    RejectSyscall.cpp
    SyscallTransform.cpp
    RejectIntrinsics.cpp
    FuzzEntry.cpp
    Cpp.cpp
    Heap.cpp
    StackSpill.cpp
    Sprintf.cpp
    Libc.cpp
    Variadic.cpp
    Reloc.cpp
    Math.cpp
    Complex.cpp
    StaticGlobals.cpp
    GlobalCtors.cpp
    MemoryLayout.cpp
    Coverage.cpp
    Asan.cpp
    ExternalSymbolGatekeeper.cpp
    AddressSpace.cpp
    Align.cpp
    LineTrace.cpp
)

add_library(CoquiPassPlugin SHARED ${COQUI_PASS_SOURCES})

target_compile_options(CoquiPassPlugin PRIVATE -fno-rtti -Wall -Wextra)
