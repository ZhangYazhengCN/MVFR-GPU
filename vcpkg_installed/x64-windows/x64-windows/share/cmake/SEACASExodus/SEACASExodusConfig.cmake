get_filename_component(VCPKG_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
# @HEADER
# ************************************************************************
#
#            TriBITS: Tribal Build, Integrate, and Test System
#                    Copyright 2013 Sandia Corporation
#
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ************************************************************************
# @HEADER

##############################################################################
#
# CMake variable for use by Seacas/SEACASExodus clients.
#
# Do not edit: This file was generated automatically by CMake.
#
##############################################################################

if(CMAKE_VERSION VERSION_LESS 3.3)
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
    "SEACASExodus requires CMake 3.3 or later for 'if (... IN_LIST ...)'"
    )
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endif()
cmake_minimum_required(VERSION 3.3...3.17.0)

## ---------------------------------------------------------------------------
## Compilers used by Seacas/SEACASExodus build
## ---------------------------------------------------------------------------

set(SEACASExodus_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe")

set(SEACASExodus_C_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe")

set(SEACASExodus_Fortran_COMPILER "")
# Deprecated!
set(SEACASExodus_FORTRAN_COMPILER "") 


## ---------------------------------------------------------------------------
## Compiler flags used by Seacas/SEACASExodus build
## ---------------------------------------------------------------------------

## Give the build type
set(SEACASExodus_CMAKE_BUILD_TYPE "Release")

## Set compiler flags, including those determined by build type
set(SEACASExodus_CXX_FLAGS [[ ]])

set(SEACASExodus_C_FLAGS [[ /nologo /DWIN32 /D_WINDOWS /utf-8 /MP  ]])

set(SEACASExodus_Fortran_FLAGS [[ ]])
# Deprecated
set(SEACASExodus_FORTRAN_FLAGS [[ ]])

## Extra link flags (e.g., specification of fortran libraries)
set(SEACASExodus_EXTRA_LD_FLAGS [[]])

## This is the command-line entry used for setting rpaths. In a build
## with static libraries it will be empty.
set(SEACASExodus_SHARED_LIB_RPATH_COMMAND "${VCPKG_IMPORT_PREFIX}/lib")
set(SEACASExodus_BUILD_SHARED_LIBS "ON")

set(SEACASExodus_LINKER C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/link.exe)
set(SEACASExodus_AR C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/lib.exe)

## ---------------------------------------------------------------------------
## Set library specifications and paths
## ---------------------------------------------------------------------------

## Base install location (if not in the build tree)
set(SEACASExodus_INSTALL_DIR "${VCPKG_IMPORT_PREFIX}")

## List of package include dirs
set(SEACASExodus_INCLUDE_DIRS "")

## List of package library paths
set(SEACASExodus_LIBRARY_DIRS "")

## List of package libraries
set(SEACASExodus_LIBRARIES "SEACASExodus::exodus")

## Specification of directories for TPL headers
set(SEACASExodus_TPL_INCLUDE_DIRS "")

## Specification of directories for TPL libraries
set(SEACASExodus_TPL_LIBRARY_DIRS "")

## List of required TPLs
set(SEACASExodus_TPL_LIBRARIES "Netcdf::all_libs;HDF5::all_libs")

## ---------------------------------------------------------------------------
## MPI specific variables
##   These variables are provided to make it easier to get the mpi libraries
##   and includes on systems that do not use the mpi wrappers for compiling
## ---------------------------------------------------------------------------

set(SEACASExodus_MPI_LIBRARIES "")
set(SEACASExodus_MPI_LIBRARY_DIRS "")
set(SEACASExodus_MPI_INCLUDE_DIRS "")
set(SEACASExodus_MPI_EXEC "")
set(SEACASExodus_MPI_EXEC_MAX_NUMPROCS "")
set(SEACASExodus_MPI_EXEC_NUMPROCS_FLAG "")

## ---------------------------------------------------------------------------
## Set useful general variables
## ---------------------------------------------------------------------------

## The packages enabled for this project
set(SEACASExodus_PACKAGE_LIST "SEACASExodus")

## The TPLs enabled for this project
set(SEACASExodus_TPL_LIST "Netcdf;HDF5")

# Enables/Disables for upstream package dependencies
set(SEACASExodus_ENABLE_Netcdf ON)
set(SEACASExodus_ENABLE_Pthread OFF)
set(SEACASExodus_ENABLE_HDF5 ON)
set(SEACASExodus_ENABLE_Pnetcdf OFF)
set(SEACASExodus_ENABLE_MPI OFF)

# Exported cache variables
set(SEACAS_ENABLE_DEBUG "OFF")
set(HAVE_SEACAS_DEBUG "OFF")
set(SEACASExodus_ENABLE_THREADSAFE "OFF")
set(EXODUS_THREADSAFE "OFF")
set(SEACASIoss_ENABLE_THREADSAFE "OFF")
set(IOSS_THREADSAFE "OFF")

# Include configuration of dependent packages
if (NOT TARGET TPL::Netcdf::all_libs)
  include("${CMAKE_CURRENT_LIST_DIR}/../../external_packages/TPL-Seacas-Netcdf/TPL-Seacas-NetcdfConfig.cmake")
endif()
if (NOT TARGET TPL::HDF5::all_libs)
  include("${CMAKE_CURRENT_LIST_DIR}/../../external_packages/TPL-Seacas-HDF5/TPL-Seacas-HDF5Config.cmake")
endif()

# Import SEACASExodus targets
include("${CMAKE_CURRENT_LIST_DIR}/SEACASExodusTargets.cmake")

## ----------------------------------------------------------------------------
## Create deprecated non-namespaced library targets for backwards compatibility
## ----------------------------------------------------------------------------

set(SEACASExodus_EXPORTED_PACKAGE_LIBS_NAMES "exodus")

foreach(libname IN LISTS SEACASExodus_EXPORTED_PACKAGE_LIBS_NAMES)
  if (NOT TARGET ${libname})
    add_library(${libname} INTERFACE IMPORTED)
    target_link_libraries(${libname}
       INTERFACE SEACASExodus::${libname})
    set(deprecationMessage
      "WARNING: The non-namespaced target '${libname}' is deprecated!"
      "  If always using newer versions of the project 'Seacas', then use the"
      " new namespaced target 'SEACASExodus::${libname}', or better yet,"
      " 'SEACASExodus::all_libs' to be less sensitive to changes in the definition"
      " of targets in the package 'SEACASExodus'.  Or, to maintain compatibility with"
      " older or newer versions the project 'Seacas', instead link against the"
      " libraries specified by the variable 'SEACASExodus_LIBRARIES'."
      )
    string(REPLACE ";" "" deprecationMessage "${deprecationMessage}")
    set_target_properties(${libname}
      PROPERTIES DEPRECATION "${deprecationMessage}" )
  endif()
endforeach()
