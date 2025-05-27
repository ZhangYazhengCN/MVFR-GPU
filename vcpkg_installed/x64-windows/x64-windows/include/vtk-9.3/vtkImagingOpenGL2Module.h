
#ifndef VTKIMAGINGOPENGL2_EXPORT_H
#define VTKIMAGINGOPENGL2_EXPORT_H

#ifdef VTKIMAGINGOPENGL2_STATIC_DEFINE
#  define VTKIMAGINGOPENGL2_EXPORT
#  define VTKIMAGINGOPENGL2_NO_EXPORT
#else
#  ifndef VTKIMAGINGOPENGL2_EXPORT
#    ifdef ImagingOpenGL2_EXPORTS
        /* We are building this library */
#      define VTKIMAGINGOPENGL2_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define VTKIMAGINGOPENGL2_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef VTKIMAGINGOPENGL2_NO_EXPORT
#    define VTKIMAGINGOPENGL2_NO_EXPORT 
#  endif
#endif

#ifndef VTKIMAGINGOPENGL2_DEPRECATED
#  define VTKIMAGINGOPENGL2_DEPRECATED __declspec(deprecated)
#endif

#ifndef VTKIMAGINGOPENGL2_DEPRECATED_EXPORT
#  define VTKIMAGINGOPENGL2_DEPRECATED_EXPORT VTKIMAGINGOPENGL2_EXPORT VTKIMAGINGOPENGL2_DEPRECATED
#endif

#ifndef VTKIMAGINGOPENGL2_DEPRECATED_NO_EXPORT
#  define VTKIMAGINGOPENGL2_DEPRECATED_NO_EXPORT VTKIMAGINGOPENGL2_NO_EXPORT VTKIMAGINGOPENGL2_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef VTKIMAGINGOPENGL2_NO_DEPRECATED
#    define VTKIMAGINGOPENGL2_NO_DEPRECATED
#  endif
#endif

/* VTK-HeaderTest-Exclude: vtkImagingOpenGL2Module.h */

/* Include ABI Namespace */
#include "vtkABINamespace.h"

#endif /* VTKIMAGINGOPENGL2_EXPORT_H */
