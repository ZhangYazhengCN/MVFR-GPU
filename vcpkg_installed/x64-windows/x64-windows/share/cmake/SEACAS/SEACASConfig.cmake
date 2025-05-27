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
# CMake variable for use by Seacas/SEACAS clients.
#
# Do not edit: This file was generated automatically by CMake.
#
##############################################################################

if(CMAKE_VERSION VERSION_LESS 3.3)
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
    "SEACAS requires CMake 3.3 or later for 'if (... IN_LIST ...)'"
    )
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endif()
cmake_minimum_required(VERSION 3.3...3.17.0)

## ---------------------------------------------------------------------------
## Compilers used by Seacas/SEACAS build
## ---------------------------------------------------------------------------

set(SEACAS_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe")

set(SEACAS_C_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe")

set(SEACAS_Fortran_COMPILER "")
# Deprecated!
set(SEACAS_FORTRAN_COMPILER "") 


## ---------------------------------------------------------------------------
## Compiler flags used by Seacas/SEACAS build
## ---------------------------------------------------------------------------

## Give the build type
set(SEACAS_CMAKE_BUILD_TYPE "Release")

## Set compiler flags, including those determined by build type
set(SEACAS_CXX_FLAGS [[ ]])

set(SEACAS_C_FLAGS [[ /nologo /DWIN32 /D_WINDOWS /utf-8 /MP  ]])

set(SEACAS_Fortran_FLAGS [[ ]])
# Deprecated
set(SEACAS_FORTRAN_FLAGS [[ ]])

## Extra link flags (e.g., specification of fortran libraries)
set(SEACAS_EXTRA_LD_FLAGS [[]])

## This is the command-line entry used for setting rpaths. In a build
## with static libraries it will be empty.
set(SEACAS_SHARED_LIB_RPATH_COMMAND "${VCPKG_IMPORT_PREFIX}/lib")
set(SEACAS_BUILD_SHARED_LIBS "ON")

set(SEACAS_LINKER C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/link.exe)
set(SEACAS_AR C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/lib.exe)

## ---------------------------------------------------------------------------
## Set library specifications and paths
## ---------------------------------------------------------------------------

## Base install location (if not in the build tree)
set(SEACAS_INSTALL_DIR "${VCPKG_IMPORT_PREFIX}")

## List of package include dirs
set(SEACAS_INCLUDE_DIRS "")

## List of package library paths
set(SEACAS_LIBRARY_DIRS "")

## List of package libraries
set(SEACAS_LIBRARIES "SEACASIoss::io_info_lib;SEACASIoss::Ionit;SEACASIoss::Iotr;SEACASIoss::Iohb;SEACASIoss::Iogs;SEACASIoss::Iotm;SEACASIoss::Iogn;SEACASIoss::Iovs;SEACASIoss::Iocgns;SEACASIoss::Ioex;SEACASIoss::Ioss;SEACASNemesis::nemesis;SEACASExodus::exodus")

## Specification of directories for TPL headers
set(SEACAS_TPL_INCLUDE_DIRS "")

## Specification of directories for TPL libraries
set(SEACAS_TPL_LIBRARY_DIRS "")

## List of required TPLs
set(SEACAS_TPL_LIBRARIES "TPL::Cereal::all_libs;fmt::all_libs;TPL::CGNS::all_libs;Netcdf::all_libs;HDF5::all_libs")

## ---------------------------------------------------------------------------
## MPI specific variables
##   These variables are provided to make it easier to get the mpi libraries
##   and includes on systems that do not use the mpi wrappers for compiling
## ---------------------------------------------------------------------------

set(SEACAS_MPI_LIBRARIES "")
set(SEACAS_MPI_LIBRARY_DIRS "")
set(SEACAS_MPI_INCLUDE_DIRS "")
set(SEACAS_MPI_EXEC "")
set(SEACAS_MPI_EXEC_MAX_NUMPROCS "")
set(SEACAS_MPI_EXEC_NUMPROCS_FLAG "")

## ---------------------------------------------------------------------------
## Set useful general variables
## ---------------------------------------------------------------------------

## The packages enabled for this project
set(SEACAS_PACKAGE_LIST "SEACASIoss;SEACASNemesis;SEACASExodus")

## The TPLs enabled for this project
set(SEACAS_TPL_LIST "Cereal;fmt;CGNS;Netcdf;HDF5")

# Enables/Disables for upstream package dependencies
set(SEACAS_ENABLE_SEACASExodus ON)
set(SEACAS_ENABLE_SEACASIoss ON)
set(SEACAS_ENABLE_SEACASExodus_for OFF)
set(SEACAS_ENABLE_SEACASExoIIv2for32 OFF)
set(SEACAS_ENABLE_SEACASNemesis ON)
set(SEACAS_ENABLE_SEACASChaco OFF)
set(SEACAS_ENABLE_SEACASAprepro_lib OFF)
set(SEACAS_ENABLE_SEACASSupes OFF)
set(SEACAS_ENABLE_SEACASSuplib OFF)
set(SEACAS_ENABLE_SEACASSuplibC OFF)
set(SEACAS_ENABLE_SEACASSuplibCpp OFF)
set(SEACAS_ENABLE_SEACASSVDI OFF)
set(SEACAS_ENABLE_SEACASPLT OFF)
set(SEACAS_ENABLE_SEACASAlgebra OFF)
set(SEACAS_ENABLE_SEACASAprepro OFF)
set(SEACAS_ENABLE_SEACASBlot OFF)
set(SEACAS_ENABLE_SEACASConjoin OFF)
set(SEACAS_ENABLE_SEACASEjoin OFF)
set(SEACAS_ENABLE_SEACASEpu OFF)
set(SEACAS_ENABLE_SEACASCpup OFF)
set(SEACAS_ENABLE_SEACASExo2mat OFF)
set(SEACAS_ENABLE_SEACASExodiff OFF)
set(SEACAS_ENABLE_SEACASExomatlab OFF)
set(SEACAS_ENABLE_SEACASExotxt OFF)
set(SEACAS_ENABLE_SEACASExo_format OFF)
set(SEACAS_ENABLE_SEACASEx1ex2v2 OFF)
set(SEACAS_ENABLE_SEACASExotec2 OFF)
set(SEACAS_ENABLE_SEACASFastq OFF)
set(SEACAS_ENABLE_SEACASGjoin OFF)
set(SEACAS_ENABLE_SEACASGen3D OFF)
set(SEACAS_ENABLE_SEACASGenshell OFF)
set(SEACAS_ENABLE_SEACASGrepos OFF)
set(SEACAS_ENABLE_SEACASExplore OFF)
set(SEACAS_ENABLE_SEACASMapvarlib OFF)
set(SEACAS_ENABLE_SEACASMapvar OFF)
set(SEACAS_ENABLE_SEACASMapvar-kd OFF)
set(SEACAS_ENABLE_SEACASMat2exo OFF)
set(SEACAS_ENABLE_SEACASNas2exo OFF)
set(SEACAS_ENABLE_SEACASZellij OFF)
set(SEACAS_ENABLE_SEACASNemslice OFF)
set(SEACAS_ENABLE_SEACASNemspread OFF)
set(SEACAS_ENABLE_SEACASNumbers OFF)
set(SEACAS_ENABLE_SEACASSlice OFF)
set(SEACAS_ENABLE_SEACASTxtexo OFF)
set(SEACAS_ENABLE_SEACASEx2ex1v2 OFF)
set(SEACAS_ENABLE_MPI OFF)

# Exported cache variables
set(SEACAS_ENABLE_DEBUG "OFF")
set(HAVE_SEACAS_DEBUG "OFF")
set(SEACASExodus_ENABLE_THREADSAFE "OFF")
set(EXODUS_THREADSAFE "OFF")
set(SEACASIoss_ENABLE_THREADSAFE "OFF")
set(IOSS_THREADSAFE "OFF")

# Include configuration of dependent packages
if (NOT TARGET SEACASExodus::all_libs)
  include("${CMAKE_CURRENT_LIST_DIR}/../SEACASExodus/SEACASExodusConfig.cmake")
endif()
if (NOT TARGET SEACASIoss::all_libs)
  include("${CMAKE_CURRENT_LIST_DIR}/../SEACASIoss/SEACASIossConfig.cmake")
endif()
if (NOT TARGET SEACASNemesis::all_libs)
  include("${CMAKE_CURRENT_LIST_DIR}/../SEACASNemesis/SEACASNemesisConfig.cmake")
endif()

# Import SEACAS targets
include("${CMAKE_CURRENT_LIST_DIR}/SEACASTargets.cmake")

## ----------------------------------------------------------------------------
## Create deprecated non-namespaced library targets for backwards compatibility
## ----------------------------------------------------------------------------

set(SEACAS_EXPORTED_PACKAGE_LIBS_NAMES "exodus;nemesis;Ioex;Iocgns;Iovs;Iogn;Iotm;Iogs;Iohb;Iotr;Ionit;io_info_lib;Ioss")

foreach(libname IN LISTS SEACAS_EXPORTED_PACKAGE_LIBS_NAMES)
  if (NOT TARGET ${libname})
    add_library(${libname} INTERFACE IMPORTED)
    target_link_libraries(${libname}
       INTERFACE SEACAS::${libname})
    set(deprecationMessage
      "WARNING: The non-namespaced target '${libname}' is deprecated!"
      "  If always using newer versions of the project 'Seacas', then use the"
      " new namespaced target 'SEACAS::${libname}', or better yet,"
      " 'SEACAS::all_libs' to be less sensitive to changes in the definition"
      " of targets in the package 'SEACAS'.  Or, to maintain compatibility with"
      " older or newer versions the project 'Seacas', instead link against the"
      " libraries specified by the variable 'SEACAS_LIBRARIES'."
      )
    string(REPLACE ";" "" deprecationMessage "${deprecationMessage}")
    set_target_properties(${libname}
      PROPERTIES DEPRECATION "${deprecationMessage}" )
  endif()
endforeach()
