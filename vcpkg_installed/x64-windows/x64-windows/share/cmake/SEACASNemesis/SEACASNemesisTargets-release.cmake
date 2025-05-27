#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SEACASNemesis::nemesis" for configuration "Release"
set_property(TARGET SEACASNemesis::nemesis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SEACASNemesis::nemesis PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/nemesis.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/nemesis.dll"
  )

list(APPEND _cmake_import_check_targets SEACASNemesis::nemesis )
list(APPEND _cmake_import_check_files_for_SEACASNemesis::nemesis "${_IMPORT_PREFIX}/lib/nemesis.lib" "${_IMPORT_PREFIX}/bin/nemesis.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
