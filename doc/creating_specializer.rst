==========================
Creating an A* Specializer
==========================

Introduction
------------
This tutorial creates a specializer for the A* Algorithm. We will guide you
through the steps you need to create your own specializer.

The A* Algorithm is used to find the shortest path between two nodes in a
graph. The A* Algorithm achieves better results by the of a heuristics
function. Depending on the application, different heuristics are used.
The idea here is that the specializer user will be able to implement his
own heuristic function in python and use it in the specializer.

Writing a specializer usually consists of the following steps:

- `Setup a project`_;
- `Create a python code to be specialized`_;
- Create tests that will be used to check the python code and later to test
  the specialized code;
- `Write specialization transformers`_ to help convert the code to C;
- Write specialization transformers to use a platform (like OpenMP, OpenCL,
  CUDA, etc.);
- Test Specializer


.. _`Setup a project`:

Starting
--------
Make sure you have ctree installed:

    ``sudo pip install ctree``

Create a specializer project with the help of the ``ctree`` command, ``-sp``
stands for *Start Project*:

    ``ctree -sp ProjectName``

A directory with the project structure will be created inside the current
directory, using the *ProjectName* you provided.

For teh A* we are calling the project *AStar*:

    ``ctree -sp AStar``


Project Files
.............
Go into the directory created, *AStar*. You will notice that all the project
structure is already created inside.

.. image:: images/project_files.png
   :width: 800px

Here is a description of each file and directory purpose:

- **AStar/** will be your project name, it is used to store the actual
  specializer, inside there are already two files: **__init__.py** and
  **main.py**;

  - **__init__.py** is used to mark the directory as a Python package, you can
    also put initialization code for your specializer package here;
  - **main.py** is where we will put the main class for the specializer, if you
    look inside the file you will see it already contains a class named *AStar*
    inherited from *LazySpecializedFunction*, we will see more about this class
    in the next sections;

- **README.rst** should contain a brief explanation about what the specializer
  do and how to use it, more detailed explanation should be placed in the doc
  subdirectory;
- **__init__.py** same purpose as the other __init__.py file;
- **cache/** will be used by ctree for cache;
- **doc/** contains the documentation files for the specializer;
- **examples/** contains examples on applications and on how to use the
  specializer;
- **setup.py** is the setup for the specializer package, contains all the
  dependencies used by the specializer;
- **templates/** contains C code templates, more details about C templates will
  be seen in the next sections;
- **tests/** contains the specializer tests, usually in the form of python
  *unittest*.


.. _`Create a python code to be specialized`:

Writing the Python Code to be Specialized
-----------------------------------------

.. note:: This section provides details on the implementation of the A*
   Algorithm in the python side. If you're in a hurry and just want to know
   about the specializer, you may skip this section and return if you need more
   details about a specific python class.

First, to implement the A* algorithm we need a priority queue. Priority queues
can be easily implemented on Python using the heapq library. An implementation
of a Python priority queue can be seen below.

.. PriorityQueue
.. include:: ../AStarSpecializer/a_star.py
   :start-line: 8
   :end-before: class Node
   :code: python

To make sure elements are ordered using the priority only, the *Node* class was
created. A *Node* object has a priority and an id but uses only the priority
for the __lt__ method.

.. Node
.. include:: ../AStarSpecializer/a_star.py
   :start-line: 27
   :end-before: class Graph
   :code: python

Now that we have the PriorityQueue set we may create the actual A* algorithm. For this we created the *Graph* class

.. Graph
.. include:: ../AStarSpecializer/a_star.py
   :start-line: 38
   :end-before: class BaseGrid
   :code: python

.. BaseGrid
.. include:: ../AStarSpecializer/a_star.py
   :start-line: 118
   :end-before: class GridAsGraph
   :code: python

.. GridAsArray
.. include:: ../AStarSpecializer/a_star.py
   :start-line: 189
   :code: python


.. _`Write specialization transformers`:

Specializing the A* code
------------------------
This section explains how to create a specializer from your python code. Here
we create the specializer for the A* Algorithm, but the steps will be similar
to what you need to do for your specializer.


Next sections should also contain: LazySpecializedFunction, C templates

