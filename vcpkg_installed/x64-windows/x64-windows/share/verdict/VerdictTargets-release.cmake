#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Verdict::verdict" for configuration "Release"
set_property(TARGET Verdict::verdict APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Verdict::verdict PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/verdict.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/verdict.dll"
  )

list(APPEND _cmake_import_check_targets Verdict::verdict )
list(APPEND _cmake_import_check_files_for_Verdict::verdict "${_IMPORT_PREFIX}/lib/verdict.lib" "${_IMPORT_PREFIX}/bin/verdict.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
