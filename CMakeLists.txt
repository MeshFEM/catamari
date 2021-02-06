cmake_minimum_required(VERSION 3.8)
project(Catamari)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

find_package(BLAS REQUIRED)
find_package(OpenMP QUIET)

if (NOT TARGET catamari)
    add_library(catamari INTERFACE)
    target_compile_definitions(catamari INTERFACE -DDARWIN -DCATAMARI_HAVE_FENV_DISABLE_DENORMS -DCATAMARI_HAVE_BLAS -DCATAMARI_HAVE_LAPACK -DQUOTIENT_USE_64BIT)
    target_link_libraries(catamari INTERFACE BLAS::BLAS)
    target_include_directories(catamari INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(catamari INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/quotient/include)
    target_include_directories(catamari INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/mantis/include)
    target_include_directories(catamari INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/specify/include)
    if (OpenMP_FOUND)
        target_link_libraries(catamari INTERFACE OpenMP::OpenMP)
    endif()
    set_target_properties(catamari PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_14)

    # trigger meson to download the subprojects
    message("running meson in ${CMAKE_CURRENT_SOURCE_DIR}")
    execute_process(COMMAND meson setup build
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
