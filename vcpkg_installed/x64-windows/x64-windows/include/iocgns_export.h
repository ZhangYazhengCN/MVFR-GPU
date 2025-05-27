
#ifndef IOCGNS_EXPORT_H
#define IOCGNS_EXPORT_H

#ifdef IOCGNS_STATIC_DEFINE
#  define IOCGNS_EXPORT
#  define IOCGNS_NO_EXPORT
#else
#  ifndef IOCGNS_EXPORT
#    ifdef Iocgns_EXPORTS
        /* We are building this library */
#      define IOCGNS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOCGNS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOCGNS_NO_EXPORT
#    define IOCGNS_NO_EXPORT 
#  endif
#endif

#ifndef IOCGNS_DEPRECATED
#  define IOCGNS_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOCGNS_DEPRECATED_EXPORT
#  define IOCGNS_DEPRECATED_EXPORT IOCGNS_EXPORT IOCGNS_DEPRECATED
#endif

#ifndef IOCGNS_DEPRECATED_NO_EXPORT
#  define IOCGNS_DEPRECATED_NO_EXPORT IOCGNS_NO_EXPORT IOCGNS_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOCGNS_NO_DEPRECATED
#    define IOCGNS_NO_DEPRECATED
#  endif
#endif

#endif /* IOCGNS_EXPORT_H */
