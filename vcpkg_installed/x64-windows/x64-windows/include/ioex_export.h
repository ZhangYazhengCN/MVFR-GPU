
#ifndef IOEX_EXPORT_H
#define IOEX_EXPORT_H

#ifdef IOEX_STATIC_DEFINE
#  define IOEX_EXPORT
#  define IOEX_NO_EXPORT
#else
#  ifndef IOEX_EXPORT
#    ifdef Ioex_EXPORTS
        /* We are building this library */
#      define IOEX_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOEX_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOEX_NO_EXPORT
#    define IOEX_NO_EXPORT 
#  endif
#endif

#ifndef IOEX_DEPRECATED
#  define IOEX_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOEX_DEPRECATED_EXPORT
#  define IOEX_DEPRECATED_EXPORT IOEX_EXPORT IOEX_DEPRECATED
#endif

#ifndef IOEX_DEPRECATED_NO_EXPORT
#  define IOEX_DEPRECATED_NO_EXPORT IOEX_NO_EXPORT IOEX_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOEX_NO_DEPRECATED
#    define IOEX_NO_DEPRECATED
#  endif
#endif

#endif /* IOEX_EXPORT_H */
