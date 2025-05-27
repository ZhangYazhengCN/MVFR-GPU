#ifndef EXODUS_CONFIG_H
#define EXODUS_CONFIG_H
/* #undef EXODUS_THREADSAFE */

#ifndef SEACAS_DEPRECATED
#  if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define SEACAS_DEPRECATED  __attribute__((__deprecated__))
#  else
#    define SEACAS_DEPRECATED
#  endif
#endif

#ifndef SEACAS_DEPRECATED_MSG
#  if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#    define SEACAS_DEPRECATED_MSG(MSG)  __attribute__((__deprecated__ (#MSG) ))
#  elif (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define SEACAS_DEPRECATED_MSG(MSG)  __attribute__((__deprecated__))
#  else
#    define SEACAS_DEPRECATED_MSG(MSG)
#  endif
#endif


#define EXODUSII_BUILD_SHARED_LIBS
#if defined(_WIN32)
# if defined(EXODUSII_BUILD_SHARED_LIBS)
#  if defined(exoIIc_EXPORTS)
#    define EXODUS_EXPORT __declspec( dllexport ) extern
#  else
#    define EXODUS_EXPORT __declspec( dllimport ) extern
#  endif
# endif
#else
#  define EXODUS_EXPORT extern
#endif

#endif
