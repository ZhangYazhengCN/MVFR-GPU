
#ifndef IOTM_EXPORT_H
#define IOTM_EXPORT_H

#ifdef IOTM_STATIC_DEFINE
#  define IOTM_EXPORT
#  define IOTM_NO_EXPORT
#else
#  ifndef IOTM_EXPORT
#    ifdef Iotm_EXPORTS
        /* We are building this library */
#      define IOTM_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOTM_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOTM_NO_EXPORT
#    define IOTM_NO_EXPORT 
#  endif
#endif

#ifndef IOTM_DEPRECATED
#  define IOTM_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOTM_DEPRECATED_EXPORT
#  define IOTM_DEPRECATED_EXPORT IOTM_EXPORT IOTM_DEPRECATED
#endif

#ifndef IOTM_DEPRECATED_NO_EXPORT
#  define IOTM_DEPRECATED_NO_EXPORT IOTM_NO_EXPORT IOTM_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOTM_NO_DEPRECATED
#    define IOTM_NO_DEPRECATED
#  endif
#endif

#endif /* IOTM_EXPORT_H */
