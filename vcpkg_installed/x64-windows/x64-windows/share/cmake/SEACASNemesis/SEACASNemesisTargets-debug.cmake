#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SEACASNemesis::nemesis" for configuration "Debug"
set_property(TARGET SEACASNemesis::nemesis APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SEACASNemesis::nemesis PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/nemesis.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/nemesis.dll"
  )

list(APPEND _cmake_import_check_targets SEACASNemesis::nemesis )
list(APPEND _cmake_import_check_files_for_SEACASNemesis::nemesis "${_IMPORT_PREFIX}/debug/lib/nemesis.lib" "${_IMPORT_PREFIX}/debug/bin/nemesis.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
