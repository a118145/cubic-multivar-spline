AI contribution
=================

The vast majority of the docstrings and some of the code in this library were partly or fully written by AI. The Claude chatbot (Sonnet 4.6) from Antropic was used. The following list provides an overview of the files or parts thereof where code was created or significantly modified by AI:

- :py:meth:`cubicmultispline.Spline1D.eval_spline`: AI-modified based on non-vectorized version 
- :py:meth:`cubicmultispline.Spline.eval_spline`: fully AI-generated as separate function, incorporation into library by human

Additionally, some private methods were created or modified by AI:

- :py:meth:`cubicmultispline.Spline1D._phi`: AI-modified based on non-vectorized version
- :py:meth:`cubicmultispline.Spline1D._dphi_dt`: AI-modified based on non-vectorized version
- :py:meth:`cubicmultispline.Spline1D._d2phi_dt2`: AI-modified based on non-vectorized version
- :py:meth:`cubicmultispline.Spline1D._d3phi_dt3`: AI-modified based on non-vectorized version
- :py:meth:`cubicmultispline.Spline._find_span_and_local`: AI-generated, incorporation into library by human

The file ``tests/test_functions.py`` was created by AI, too. It holds two test functions with analytical gradients and Hessians for testing the spline interpolation. 