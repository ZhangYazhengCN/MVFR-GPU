#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CGNS::cgns_shared" for configuration "Release"
set_property(TARGET CGNS::cgns_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(CGNS::cgns_shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/cgnsdll.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/cgnsdll.dll"
  )

list(APPEND _cmake_import_check_targets CGNS::cgns_shared )
list(APPEND _cmake_import_check_files_for_CGNS::cgns_shared "${_IMPORT_PREFIX}/lib/cgnsdll.lib" "${_IMPORT_PREFIX}/bin/cgnsdll.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
