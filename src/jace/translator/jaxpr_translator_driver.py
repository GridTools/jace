# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


class JaxprTranslationDriver:
    """Internal driver class for creating an SDFG equivalent of a `Jaxpr` instance.

    This class builds an SDFG of a very particular form, which for us is
    canonical, which is not directly usable. Thus this class should not be
    directly used, instead a user should use TBA.
    The canonical form is characterized by the following:
    - the SDFG is a list of states, ideally each state corresponds to single Jax primitive,
    - all variable names are derived from Jax names,
    - there are no global variables inside the SDFG,
    - It lacks the special `__return` variable.
    - The argument names are not set.

    The idea of the translator is extremely simple. Since Jaxpr is a list
    consisting of more or less simple instructions/equations, they get processed
    one after the other. Each equation is translated into its own state that
    is appended to the SDFG, thus the SDFG is a long list of states. In certain
    cases it might be that an equation needs more states, but this is an exception.

    The actual translation is not handled by the driver instead a so called
    subtranslator object is used. A subtranslator is specialized to translate
    one type of primitive. For more information on the subtranslators see the
    documentation of `JaCeSubTranslatorInterface`.

    To support nested Jaxpr expressions the driver provides the possibility to
    clone/fork itself, see `self.fork()` for more. Every clone, i.e. return
    value of `self.fork()`, of a driver, which is also known as child, has
    a unique identifier. This identifier is used for example to generate
    unique SDFG variable names during a translation process,
    see `self.same_family() for more.

    If no translation is ongoing the only function that makes sense to call
    is `translate_jaxpr()` which starts a translation.

    Todos:
        Find a better way than to allow giving access to protected functions.
            Probably using composition with the higher level instance.
    """
