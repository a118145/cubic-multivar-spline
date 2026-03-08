Examples
========

The examples in this section demonstrate detailed use cases of the library. Apart from increasing dimensional complexity, the examples also serve as templates for the user to adapt to their own use case. All examples can be found in the tests directory of the `repository`_.

.. _repository: https://github.com/a118145/cubic-multivar-spline/

.. admonition:: :ref:`Quickstart: 2D example -- scalar field with uniform boundary conditions <quickstart>`
   :class: information

   * ``tests/demo_not-a-knot.py``
   * 2D interpolation of a scalar field of random, equidistant samples with not-a-knot boundary conditions
   * Visual inspection of interpolation 

.. admonition:: :ref:`2D example -- scalar field <boundary2d>`
   :class: note

   * ``tests/demo_first-second-peri.py``
   * 2D interpolation of a scalar field of random, equidistant samples with different boundary conditions in each dimension
   * Inspection of boundary conditions by means of 1D slices along the edges of the domain

.. admonition:: :ref:`3D example -- scalar field <boundary3d>`
   :class: note

   * ``tests/demo_first-second-3d.py``
   * 3D interpolation of a scalar field of random, equidistant samples with different boundary conditions in each dimension
   * Inspection of boundary conditions by means of 2D slices along the edges of the domain
   * Animation of stacked slices to visualize 3-dimensional interpolation

.. toctree::
   :hidden:
   
   boundary2d
   boundary3d