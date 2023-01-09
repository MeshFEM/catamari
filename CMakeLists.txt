cmake_minimum_required(VERSION 3.16)
project(Catamari)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(BLA_VENDOR Apple)
else()
set(BLA_VENDOR Intel10_64_dyn)
endif()

find_package(BLAS REQUIRED)

if (NOT TARGET catamari)
    add_library(catamari INTERFACE)
    # TODO: actually detect the proper flags here rather than hard-coding for
    # M1-based Mac and x86-based Linux.
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        target_compile_definitions(catamari INTERFACE -DDARWIN -DARM -DCATAMARI_HAVE_FENV_DISABLE_DENORMS -DCATAMARI_HAVE_BLAS -DCATAMARI_HAVE_LAPACK -DQUOTIENT_USE_64BIT)
    elseif  (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        target_compile_definitions(catamari INTERFACE -DMANTIS_GCC_FUSED_MULTIPLY_ADD -DX86 -DLINUX -DCATAMARI_HAVE_XMMINTRIN -DMANTIS_GCC_FUSED_MULTIPLY_ADD -DCATAMARI_HAVE_MKL -DCATAMARI_HAVE_BLAS -DCATAMARI_HAVE_LAPACK -DQUOTIENT_USE_64BIT)
    else()
        message(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME")
    endif()
    target_link_libraries(catamari INTERFACE BLAS::BLAS)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/quotient/include)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/mantis/include)
    target_include_directories(catamari SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/specify/include)
    set_target_properties(catamari PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_14)

    # trigger meson to download the subprojects
    message("running meson in ${CMAKE_CURRENT_SOURCE_DIR}")
    set(OLD_CXX "$ENV{CXX}")
    set(ENV{CXX} "")
    execute_process(COMMAND meson setup build
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE ret)
    if (ret AND NOT ret EQUAL 0)
        message(FATAL_ERROR "Meson build failed -- is meson installed?")
    endif()
    set(ENV{CXX} ${OLD_CXX})
endif()
