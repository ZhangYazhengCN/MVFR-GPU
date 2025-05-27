
#ifndef IOGS_EXPORT_H
#define IOGS_EXPORT_H

#ifdef IOGS_STATIC_DEFINE
#  define IOGS_EXPORT
#  define IOGS_NO_EXPORT
#else
#  ifndef IOGS_EXPORT
#    ifdef Iogs_EXPORTS
        /* We are building this library */
#      define IOGS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOGS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOGS_NO_EXPORT
#    define IOGS_NO_EXPORT 
#  endif
#endif

#ifndef IOGS_DEPRECATED
#  define IOGS_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOGS_DEPRECATED_EXPORT
#  define IOGS_DEPRECATED_EXPORT IOGS_EXPORT IOGS_DEPRECATED
#endif

#ifndef IOGS_DEPRECATED_NO_EXPORT
#  define IOGS_DEPRECATED_NO_EXPORT IOGS_NO_EXPORT IOGS_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOGS_NO_DEPRECATED
#    define IOGS_NO_DEPRECATED
#  endif
#endif

#endif /* IOGS_EXPORT_H */
