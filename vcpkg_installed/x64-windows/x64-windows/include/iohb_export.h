
#ifndef IOHB_EXPORT_H
#define IOHB_EXPORT_H

#ifdef IOHB_STATIC_DEFINE
#  define IOHB_EXPORT
#  define IOHB_NO_EXPORT
#else
#  ifndef IOHB_EXPORT
#    ifdef Iohb_EXPORTS
        /* We are building this library */
#      define IOHB_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOHB_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOHB_NO_EXPORT
#    define IOHB_NO_EXPORT 
#  endif
#endif

#ifndef IOHB_DEPRECATED
#  define IOHB_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOHB_DEPRECATED_EXPORT
#  define IOHB_DEPRECATED_EXPORT IOHB_EXPORT IOHB_DEPRECATED
#endif

#ifndef IOHB_DEPRECATED_NO_EXPORT
#  define IOHB_DEPRECATED_NO_EXPORT IOHB_NO_EXPORT IOHB_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOHB_NO_DEPRECATED
#    define IOHB_NO_DEPRECATED
#  endif
#endif

#endif /* IOHB_EXPORT_H */
