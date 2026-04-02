# FindOptiX.cmake for OptiX 7+/9+ (header-only SDK)
#
# Sets:
#   OptiX_FOUND
#   OptiX_INCLUDE - path containing optix.h
#
# OptiX 7+ is header-only. The runtime loads the driver's OptiX
# implementation via optix_stubs.h — no library to link.

set(OptiX_INSTALL_DIR "/opt/optix" CACHE PATH "Path to OptiX SDK")

find_path(OptiX_INCLUDE
    NAMES optix.h
    PATHS
        "${OptiX_INSTALL_DIR}/include"
        "$ENV{OPTIX_PATH}/include"
        "$ENV{OptiX_INSTALL_DIR}/include"
    NO_DEFAULT_PATH
)

if(OptiX_INCLUDE)
    set(OptiX_FOUND TRUE)
    message(STATUS "Found OptiX: ${OptiX_INCLUDE}")
else()
    if(OptiX_FIND_REQUIRED)
        message(FATAL_ERROR "OptiX headers not found. Set OptiX_INSTALL_DIR.")
    else()
        message(STATUS "OptiX not found (optional)")
    endif()
endif()
