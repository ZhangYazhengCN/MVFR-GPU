
#ifndef IOGN_EXPORT_H
#define IOGN_EXPORT_H

#ifdef IOGN_STATIC_DEFINE
#  define IOGN_EXPORT
#  define IOGN_NO_EXPORT
#else
#  ifndef IOGN_EXPORT
#    ifdef Iogn_EXPORTS
        /* We are building this library */
#      define IOGN_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IOGN_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IOGN_NO_EXPORT
#    define IOGN_NO_EXPORT 
#  endif
#endif

#ifndef IOGN_DEPRECATED
#  define IOGN_DEPRECATED __declspec(deprecated)
#endif

#ifndef IOGN_DEPRECATED_EXPORT
#  define IOGN_DEPRECATED_EXPORT IOGN_EXPORT IOGN_DEPRECATED
#endif

#ifndef IOGN_DEPRECATED_NO_EXPORT
#  define IOGN_DEPRECATED_NO_EXPORT IOGN_NO_EXPORT IOGN_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IOGN_NO_DEPRECATED
#    define IOGN_NO_DEPRECATED
#  endif
#endif

#endif /* IOGN_EXPORT_H */
