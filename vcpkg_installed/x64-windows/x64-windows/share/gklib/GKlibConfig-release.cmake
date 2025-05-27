#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GKlib" for configuration "Release"
set_property(TARGET GKlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GKlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/GKlib.lib"
  )

list(APPEND _cmake_import_check_targets GKlib )
list(APPEND _cmake_import_check_files_for_GKlib "${_IMPORT_PREFIX}/lib/GKlib.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
