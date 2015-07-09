==========================
Creating an A* Specializer
==========================

Introduction
------------
This tutorial creates a specializer for the A* Algorithm. We will guide you
through the steps you need to create your own specializer.

The A* Algorithm is used to find the shortest path between two nodes in a
graph. This algorithm achieves better results by the of a heuristics
function. Depending on the application, different heuristics are used.
The idea here is that the specializer user will be able to implement his
own heuristic function in python and use it in the specializer.

Writing a specializer usually consists of the following steps:

- `Setup a project`_;
- `Writing the Python Code to be Specialized`_;
- Create tests that will be used to check the python code and later to test
  the specialized code;
- `Write specialization transformers`_ to help convert the code to C;
- Write specialization transformers to use a platform (like OpenMP, OpenCL,
  CUDA, etc.);
- Test Specializer


Basic Concepts
..............

- **AST** Abstract Syntax Tree. A tree representation of a source code. This is
  the way specializers modify and convert codes.
- **JIT** Just in Time. Refers to the "just in time" compilation of the code.
  A specializer JIT compiles (compiles just in time) part of the python code
  specialized.
- **Transformer** Same as Visitor with the difference that Transformers can
  modify the tree they are traversing.
- **Visitor** A class that traverses a tree and executes actions based on the
  values of specific types of node, but without modifying them.


Setup a project
---------------
Make sure you have ctree installed:

    ``sudo pip install ctree``

Create a specializer project with the help of the ``ctree`` command, ``-sp``
stands for *Start Project*:

    ``ctree -sp ProjectName``

A directory with the project structure will be created inside the current
directory, using the *ProjectName* you provided.

For the A* we are calling the project *AStar*:

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

Now that we have the PriorityQueue set we may create the actual A* algorithm.
For this we created the *Graph* class

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
to what you need to do for your own specializer.

The Structure of a Specializer
..............................

A Specializer has two main classes, one inherited from the
``LazySpecializedFunction`` and the other inherited from the
``ConcreteSpecializedFunction``.

The ``LazySpecializedFunction`` will wait until the last moment before
specializing the function. You may wonder why to do that, it turns out that,
for a specializer, the last moment can be the best moment. Since you're
specializing in the last moment, your specialized function may be
created for a specific parameter type and also tuned for a specific data
structure size. For the A* specializer, the specialized function will change
according to the grid dimensions and the type of the elements.

The ``ConcreteSpecializedFunction`` is the already specialized and compiled
function.

To start, lets create a really simple specializer.

Fibonacci Specializer
`````````````````````
We will start by creating the fibonacci function in python.

.. code:: python

    def fib(n):
        if n < 2:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

That's the function we will specialize. To do it, we will write the two
required classes. The first can be seen below.

.. code:: python

    from ctree.types import get_ctype
    from ctree.nodes import Project
    from ctree.c.nodes import FunctionDecl, CFile
    from ctree.transformations import PyBasicConversions
    from ctree.jit import LazySpecializedFunction

    class BasicTranslator(LazySpecializedFunction):

        def args_to_subconfig(self, args):
            return {'arg_type': type(get_ctype(args[0]))}

        def transform(self, tree, program_config):
            tree = PyBasicConversions().visit(tree)

            fib_fn = tree.find(FunctionDecl, name="apply")
            arg_type = program_config.args_subconfig['arg_type']
            fib_fn.return_type = arg_type()
            fib_fn.params[0].type = arg_type()
            c_translator = CFile("generated", [tree])

            return [c_translator]

        def finalize(self, transform_result, program_config):
            proj = Project(transform_result)

            arg_config, tuner_config = program_config
            arg_type = arg_config['arg_type']
            entry_type = ctypes.CFUNCTYPE(arg_type, arg_type)

            return BasicFunction("apply", proj, entry_type)

Observe the ``BasicTranslator`` is inherited from ``LazySpecializedFunction``
class. To use the ``LazySpecializedFunction`` we import it from ``ctree.jit``.
We are overriding three methods:

.. _`args_to_subconfig`:

- **args_to_subconfig** This method receives the arguments that are being
  passed to the function we are specializing (``fib``). What is returned from
  this method will be placed on the ``program_config`` parameter passed to the
  transform_ method. This is very important as the ``program_config`` is what
  determines if a new specialized function must be created or if an already
  existing one can be used.

  Observe we return a dictionary that contains the type of the first argument
  passed to the function. When we call the ``fib`` function from python using
  an integer argument, the returned dictionary will contain the type integer.
  If we call the function again with another integer it knows it was already
  specialized for the integer type and will use the cached version. In the
  other hand, if we call ``fib`` with a different type, this will be detected
  and a new specialized function for this type will be created. Also observe
  that, to get the type, we used two functions: ``type`` and ``get_ctype``.
  ``type`` is a built-in python function to get the type. ``get_ctype`` can be
  found on ``ctree.types`` and returns the closest C type instance
  corresponding to the object. You need to use both functions.

.. _transform:

- **transform** Here is where the function transformations happen. This method
  has two parameters: ``tree`` and ``program_config``. ``tree`` is the function
  we are specializing converted to AST. ``program_config`` is a ``namedtuple``
  with two fields:

  - ``args_subconfig`` the dictionary returned by the `args_to_subconfig`_
    method;
  - ``tuner_subconfig`` contains tuning specific configurations. We are not
    using tuner here.

  For this very simple specializer we are using only a single transformer, the
  ``PyBasicConversions`` transformer. This transformer converts python code
  with obvious C analogues to C, you can import it from
  ``ctree.transformations``. It's important to notice the way the transformer
  is used. We instantiate the transformer class and then call the visit method
  passing the AST. This is the way most transformers are used. Since we only
  have a simple python code with obvious C analogues, this transformation is
  enough to transform the entire function to C.

  Next step is to convert the function return and parameters to C. The function
  we are specializing (``fib``) has its name automatically changed to ``apply``
  when being converted to AST. We can easily find the function we're
  specializing by looking for the ``apply`` function in the AST. We do this
  with the ``find`` method. In the line
  ``tree.find(FunctionDecl, name="apply")`` we're looking for a node with type
  ``FunctionDecl`` that has an attribute ``name`` with the string ``"apply"``,
  which is our function. We know the parameter type already as we got it in the
  `args_to_subconfig`_ method. For this function, the type of the parameter
  will be the same as the return. This is what we do in the following lines:
  get the parameter type from the program_config, attribute this type to the
  function ``return_type`` and to the first parameter of the function. One
  thing that may be tricky is that the ``arg_type`` we got is of *type*
  ``type`` while the function return and parameters we're assigning need an
  *instance* of this type, not the type itself. That is the reason we use
  parenthesis after ``arg_type`` when assigning the return and parameter type.

  The last step in the ``transform`` method is to put the tree in a ``CFile``,
  this is a node that represents a ``.c`` file and is what the ``transform``
  method should return. We give the ``CFile`` the name "generated" and pass the
  tree we generated to it. A list containing the ``CFile`` is finally returned.

.. _finalize:

- **finalize** This is the last thing done by the ``LazySpecializedFunction``.
  This method has two parameters: ``transform_result`` and ``program_config``.
  ``transform_result`` is what was returned by the ``transform``, the list with
  the ``CFile`` we created. ``program_config`` is the same parameter as in the
  ``transform`` method. The ``finalize`` is responsible to return a
  ``ConcreteSpecializedFunction``. The code for BasicFunction_, the class that
  inherits from ``ConcreteSpecializedFunction`` will be seen below but it
  requires a ``Project`` and an entry type. A ``Project`` is used to pack all
  the CFiles in your project, in this case just one. The entry type is the
  interface between python and the C function created.

  The ``Project`` class can be imported from ``ctree.nodes`` and it can be used
  as shown in the example, using the list of ``CFile`` as argument. To create
  the entry type we need to use the function ``CFUNCTYPE`` from the module
  ``ctypes``. The first parameter of this function is the return type, the
  following parameters are the parameter types.

.. _BasicFunction:

The implementation fo the ``BasicFunction`` is simple, we need to implement two
methods: ``__init__`` and ``__call__``. The ``__init__`` receive all the
arguments we saw in the finalize_ method and assigns a compiled function to a
function attribute.

.. code:: python

    from ctree.jit import ConcreteSpecializedFunction

    class BasicFunction(ConcreteSpecializedFunction):
        def __init__(self, entry_name, project_node, entry_typesig):
            self._c_function = self._compile(entry_name, project_node, entry_typesig)

        def __call__(self, *args, **kwargs):
            return self._c_function(*args, **kwargs)

.. code:: python

    """
    specializer AStar
    """

    from ctree.jit import LazySpecializedFunction


    class AStar(LazySpecializedFunction):

        def transform(self):
            pass

    if __name__ == '__main__':
        pass


Next sections should also contain: LazySpecializedFunction, C templates
