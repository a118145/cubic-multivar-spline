Changelog
===========

0.1.2
-----
- Added optional PyTorch backend: `TorchSpline1D` and `TorchSpline` — GPU/MPS-compatible drop-in replacements for `Spline1D` and `Spline`; install with `pip install cubicmultispline[torch]`
- Device and dtype follow the input tensor automatically; supports float32 and float64
- Refactored test suite: migrated from custom runner to pytest with `pytest.ini` config
- Added multi-dimensional accuracy tests (1D–5D) and boundary condition tests
- Added performance benchmarks (construction and evaluation, 1D–3D)
- Docu updated
- Added 3D example with first/second derivative boundary conditions
- Fixed boundary condition handling


0.1.1
-----
- Fixed import issues which led to malfunctioning library
- Tidied up files
- Moved documentation to <a href="https://cubicmultispline.readthedocs.io" target="_blank">https://cubicmultispline.readthedocs.io</a>
- Shortened README.md

0.1.0
-----
- Initial release
