
#ifndef IOTR_EXPORT_H
#define IOTR_EXPORT_H

#ifdef IOTR_STATIC_DEFINE
#  define IOTR_EXPORT
#  define IOTR_NO_EXPORT
#else
#  ifndef IOTR_EXPORT
#    ifdef Iotr_EXPORTS
        /* We are building this library */
#      define IOTR_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOTR_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOTR_NO_EXPORT
#    define IOTR_NO_EXPORT 
#  endif
#endif

#ifndef IOTR_DEPRECATED
#  define IOTR_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOTR_DEPRECATED_EXPORT
#  define IOTR_DEPRECATED_EXPORT IOTR_EXPORT IOTR_DEPRECATED
#endif

#ifndef IOTR_DEPRECATED_NO_EXPORT
#  define IOTR_DEPRECATED_NO_EXPORT IOTR_NO_EXPORT IOTR_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOTR_NO_DEPRECATED
#    define IOTR_NO_DEPRECATED
#  endif
#endif

#endif /* IOTR_EXPORT_H */
