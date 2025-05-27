#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "flann::flann_cpp" for configuration "Debug"
set_property(TARGET flann::flann_cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(flann::flann_cpp PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/flann_cppd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/flann_cppd.dll"
  )

list(APPEND _cmake_import_check_targets flann::flann_cpp )
list(APPEND _cmake_import_check_files_for_flann::flann_cpp "${_IMPORT_PREFIX}/debug/lib/flann_cppd.lib" "${_IMPORT_PREFIX}/debug/bin/flann_cppd.dll" )

# Import target "flann::flann" for configuration "Debug"
set_property(TARGET flann::flann APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(flann::flann PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/flannd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/flannd.dll"
  )

list(APPEND _cmake_import_check_targets flann::flann )
list(APPEND _cmake_import_check_files_for_flann::flann "${_IMPORT_PREFIX}/debug/lib/flannd.lib" "${_IMPORT_PREFIX}/debug/bin/flannd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
