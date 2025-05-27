
#ifndef IONIT_EXPORT_H
#define IONIT_EXPORT_H

#ifdef IONIT_STATIC_DEFINE
#  define IONIT_EXPORT
#  define IONIT_NO_EXPORT
#else
#  ifndef IONIT_EXPORT
#    ifdef Ionit_EXPORTS
        /* We are building this library */
#      define IONIT_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define IONIT_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef IONIT_NO_EXPORT
#    define IONIT_NO_EXPORT 
#  endif
#endif

#ifndef IONIT_DEPRECATED
#  define IONIT_DEPRECATED __declspec(deprecated)
#endif

#ifndef IONIT_DEPRECATED_EXPORT
#  define IONIT_DEPRECATED_EXPORT IONIT_EXPORT IONIT_DEPRECATED
#endif

#ifndef IONIT_DEPRECATED_NO_EXPORT
#  define IONIT_DEPRECATED_NO_EXPORT IONIT_NO_EXPORT IONIT_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IONIT_NO_DEPRECATED
#    define IONIT_NO_DEPRECATED
#  endif
#endif

#endif /* IONIT_EXPORT_H */
