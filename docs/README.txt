0. install requirements
   $ pip install ../rtd_requirements-docs.txt

1. Run
   $ generate_apidoc.py
   to generate `tmp_apidoc`. Compare with these in `source/apidoc`, and move
   `tmp_apidoc` to `source_apidoc`. If everything looks good.

2. To generate docs:
   generating sphinx-gallery examples (pop `source/auto_examples` directory)
   $ make html

   or

   not generating sphinx-gallery examples (much faster)
   $ make html-notutorial

3. To view locally, open ./build/html/index.html


Check the tutorials. Stdout and stderr sometimes are not included in the rst files
generated by sphinx-gallery. In this case, delete the generated file, and run it one by
one will work, e.g.
$ cd kliff/docs
$ rm source/auto_examples/example_kim_SW_Si*
$ make html
and do it for other examples as well.