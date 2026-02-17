cmake_minimum_required(VERSION 3.14)
project(lance_metal LANGUAGES CXX OBJCXX)

find_library(METAL_FW Metal REQUIRED)
find_library(FOUNDATION_FW Foundation REQUIRED)

# Compile Metal shaders to default.metallib
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_SOURCES
    ${SHADER_DIR}/l2_distance.metal
    ${SHADER_DIR}/cosine_distance.metal
    ${SHADER_DIR}/inner_product.metal
)

execute_process(
    COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(METALLIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

# Compile each .metal to .air, then link to .metallib
set(AIR_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air)
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal -c ${SHADER} -o ${AIR_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME}.metal"
    )
    list(APPEND AIR_FILES ${AIR_FILE})
endforeach()

add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking Metal shaders into default.metallib"
)

add_custom_target(lance_metal_shaders DEPENDS ${METALLIB_OUTPUT})

# Build the lance_metal library
add_library(lance_metal STATIC
    src/MetalContext.mm
    src/MetalVectorDistance.mm
)

add_dependencies(lance_metal lance_metal_shaders)

target_include_directories(lance_metal PUBLIC include)
target_link_libraries(lance_metal ${METAL_FW} ${FOUNDATION_FW})

# Embed metallib path
target_compile_definitions(lance_metal PRIVATE
    LANCE_METALLIB_PATH="${METALLIB_OUTPUT}"
)

if(MACOS_SDK_PATH)
    target_compile_options(lance_metal PRIVATE
        -iframework ${MACOS_SDK_PATH}/System/Library/Frameworks)
endif()
