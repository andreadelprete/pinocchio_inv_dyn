#
# Copyright (c) 2017 CNRS
#
# This file is part of pinocchio_inv_dyn
# pinocchio is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# pinocchio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# pinocchio If not, see
# <http://www.gnu.org/licenses/>.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/python.cmake)

SET(PROJECT_NAME pinocchio_inv_dyn)
SET(PROJECT_DESCRIPTION "A simple simulation/control environment for multi-body systems")
SET(PROJECT_URL "http://github.com/andreadelprete/pinocchio_inv_dyn")
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" FALSE)

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()

FINDPYTHON()

INSTALL(DIRECTORY python/pinocchio_inv_dyn
         DESTINATION ${PYTHON_SITELIB}/)

SETUP_PROJECT_FINALIZE()
