#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CGNS::cgns_shared" for configuration "Debug"
set_property(TARGET CGNS::cgns_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(CGNS::cgns_shared PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/cgnsdll.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/cgnsdll.dll"
  )

list(APPEND _cmake_import_check_targets CGNS::cgns_shared )
list(APPEND _cmake_import_check_files_for_CGNS::cgns_shared "${_IMPORT_PREFIX}/debug/lib/cgnsdll.lib" "${_IMPORT_PREFIX}/debug/bin/cgnsdll.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
