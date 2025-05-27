
#ifndef IOVS_EXPORT_H
#define IOVS_EXPORT_H

#ifdef IOVS_STATIC_DEFINE
#  define IOVS_EXPORT
#  define IOVS_NO_EXPORT
#else
#  ifndef IOVS_EXPORT
#    ifdef Iovs_EXPORTS
        /* We are building this library */
#      define IOVS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOVS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOVS_NO_EXPORT
#    define IOVS_NO_EXPORT 
#  endif
#endif

#ifndef IOVS_DEPRECATED
#  define IOVS_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOVS_DEPRECATED_EXPORT
#  define IOVS_DEPRECATED_EXPORT IOVS_EXPORT IOVS_DEPRECATED
#endif

#ifndef IOVS_DEPRECATED_NO_EXPORT
#  define IOVS_DEPRECATED_NO_EXPORT IOVS_NO_EXPORT IOVS_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOVS_NO_DEPRECATED
#    define IOVS_NO_DEPRECATED
#  endif
#endif

#endif /* IOVS_EXPORT_H */
