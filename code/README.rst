pyqalm
======

pyqalm: Clustering with learned fast transforms

Developper notes
----------------

Logging
+++++++

Use logger for debugging:

	from pyqalm.utils import logger

	logger.debug("your logging message level debug")
	logger.info("your logging message level info")
	logger.warning("your logging message level warning")
	logger.error("your logging message level error")

If you want to change the logging level, do:

	import logging
	import daiquiri

	daiquiri.setup(level=logging.DEBUG) # for level debug, change value of level according to your needs

	# your script (with no prints)

Status
------
This documentation is not up to date:

* Installation with ``pip`` is not available yet (from pypi repo)
* Documentation is not online yet
* Gitlab project is not public

Todos
-----

* Installation with check environment

Install
-------

Currently, install with ``pip`` from a git copy by running the following
command in directory ``[root]/qalm_qmeans/``::

    make install

!!ALL WHAT REMAINS IS NOT UP TO DATE!!

Install the current release with ``pip``::

    pip install pyqalm

For additional details, see doc/install.rst.

Usage
-----

See the `documentation <http://qarma.pages.lis-lab.fr/qarma/pyqalm/>`_.

Bugs
----

Please report any bugs that you find through the `pyqalm GitLab project
<https://gitlab.lis-lab.fr/qarma/pyqalm/issues>`_.

You can also fork the repository and create a merge request.

Source code
-----------

The source code of yafe is available via its `GitLab project
<https://gitlab.lis-lab.fr/qarma/pyqalm>`_.

You can clone the git repository of the project using the command::

    git clone git@gitlab.lis-lab.fr:qarma/pyqalm.git

Copyright © 2018-2019
---------------------

* `Laboratoire d'Informatique et Systèmes <http://www.lis-lab.fr/>`_
* `Université d'Aix-Marseille <http://www.univ-amu.fr/>`_
* `Centre National de la Recherche Scientifique <http://www.cnrs.fr/>`_
* `Université de Toulon <http://www.univ-tln.fr/>`_

Contributors
------------

* `Valentin Emiya <mailto:valentin.emiya@lis-lab.fr>`_
* `Luc Giffon <mailto:luc.giffon@lis-lab.fr>`_

License
-------

Released under the GNU General Public License version 3 or later
(see `LICENSE.txt`).
