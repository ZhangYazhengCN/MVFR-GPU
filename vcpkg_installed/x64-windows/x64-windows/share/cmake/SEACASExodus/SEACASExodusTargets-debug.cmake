#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SEACASExodus::exodus" for configuration "Debug"
set_property(TARGET SEACASExodus::exodus APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SEACASExodus::exodus PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/exodus.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/exodus.dll"
  )

list(APPEND _cmake_import_check_targets SEACASExodus::exodus )
list(APPEND _cmake_import_check_files_for_SEACASExodus::exodus "${_IMPORT_PREFIX}/debug/lib/exodus.lib" "${_IMPORT_PREFIX}/debug/bin/exodus.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
