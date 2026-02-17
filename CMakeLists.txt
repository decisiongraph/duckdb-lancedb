cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME lancedb)

# ========================================
# 1. Rust LanceDB library (staticlib via cargo)
# ========================================

find_program(CARGO_EXECUTABLE cargo)

if(CARGO_EXECUTABLE)
    message(STATUS "Rust toolchain found: ${CARGO_EXECUTABLE}")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(RUST_BUILD_TYPE "debug")
        set(RUST_BUILD_FLAG "")
    else()
        set(RUST_BUILD_TYPE "release")
        set(RUST_BUILD_FLAG "--release")
    endif()

    # Detect Rust target
    set(RUST_TARGET "")
    if("${OS_NAME}" STREQUAL "linux")
        if("${OS_ARCH}" STREQUAL "arm64" OR "${CMAKE_CXX_COMPILER}" MATCHES "aarch64")
            set(RUST_TARGET "aarch64-unknown-linux-gnu")
        else()
            set(RUST_TARGET "x86_64-unknown-linux-gnu")
        endif()
    elseif("${OS_NAME}" STREQUAL "osx" OR APPLE)
        if("${OSX_BUILD_ARCH}" STREQUAL "arm64" OR "${OS_ARCH}" STREQUAL "arm64"
           OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(RUST_TARGET "aarch64-apple-darwin")
        else()
            set(RUST_TARGET "x86_64-apple-darwin")
        endif()
    elseif(WIN32)
        if("${OS_ARCH}" STREQUAL "arm64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ARM64")
            set(RUST_TARGET "aarch64-pc-windows-msvc")
        else()
            set(RUST_TARGET "x86_64-pc-windows-msvc")
        endif()
    endif()

    if(NOT "${RUST_TARGET}" STREQUAL "")
        set(RUST_TARGET_FLAG "--target=${RUST_TARGET}")
        set(RUST_TARGET_DIR "${RUST_TARGET}/")
        message(STATUS "Rust target: ${RUST_TARGET}")
    else()
        set(RUST_TARGET_FLAG "")
        set(RUST_TARGET_DIR "")
    endif()

    if(WIN32)
        set(RUST_LIB_NAME "lancedb_rust.lib")
    else()
        set(RUST_LIB_NAME "liblancedb_rust.a")
    endif()

    set(RUST_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust_lib)
    set(RUST_LIB_PATH ${RUST_LIB_DIR}/target/${RUST_TARGET_DIR}${RUST_BUILD_TYPE}/${RUST_LIB_NAME})

    add_custom_command(
        OUTPUT ${RUST_LIB_PATH}
        COMMAND ${CARGO_EXECUTABLE} build ${RUST_BUILD_FLAG} ${RUST_TARGET_FLAG}
        WORKING_DIRECTORY ${RUST_LIB_DIR}
        COMMENT "Building Rust LanceDB library..."
        DEPENDS
            ${RUST_LIB_DIR}/Cargo.toml
            ${RUST_LIB_DIR}/src/lib.rs
            ${RUST_LIB_DIR}/src/ffi.rs
            ${RUST_LIB_DIR}/src/lance_manager.rs
            ${RUST_LIB_DIR}/src/runtime.rs
    )

    add_custom_target(lancedb_rust_build DEPENDS ${RUST_LIB_PATH})

    add_library(lancedb_rust STATIC IMPORTED GLOBAL)
    set_target_properties(lancedb_rust PROPERTIES
        IMPORTED_LOCATION ${RUST_LIB_PATH}
    )
    add_dependencies(lancedb_rust lancedb_rust_build)

    set(LANCEDB_RUST_AVAILABLE ON)
    message(STATUS "Rust LanceDB: ENABLED")
else()
    message(FATAL_ERROR "Rust toolchain not found - LanceDB requires Rust")
endif()

# ========================================
# 2. C++ Extension sources
# ========================================

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES
    src/lancedb_extension.cpp
    src/lance_index.cpp
    src/lance_optimizer.cpp
    src/lance_search.cpp
    src/lance_functions.cpp
    src/lance_list.cpp
    src/rust_ffi.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# ========================================
# 3. Linking
# ========================================

add_dependencies(${EXTENSION_NAME} lancedb_rust_build)
add_dependencies(${LOADABLE_EXTENSION_NAME} lancedb_rust_build)

target_link_libraries(${EXTENSION_NAME} ${RUST_LIB_PATH})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${RUST_LIB_PATH})

# Platform-specific system libraries for Rust
if(APPLE)
    target_link_libraries(${EXTENSION_NAME}
        "-framework Security" "-framework CoreFoundation" "-framework SystemConfiguration")
    target_link_libraries(${LOADABLE_EXTENSION_NAME}
        "-framework Security" "-framework CoreFoundation" "-framework SystemConfiguration")
elseif(WIN32)
    target_link_libraries(${EXTENSION_NAME} ws2_32 userenv bcrypt ntdll)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} ws2_32 userenv bcrypt ntdll)
elseif(UNIX)
    target_link_libraries(${EXTENSION_NAME} dl pthread m)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} dl pthread m)
endif()

install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
