# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals

import logging

from pymatgen import MPRester, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
from pymatgen.analysis.reaction_calculator import Reaction
import numpy as np
from anytree import Node, RenderTree


__author__ = "Alex Dunn, Anubhav Jain, Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"


class PDSynthesisTree:

    """
    This class generates a synthesis tree using a phase diagram analysis.
    Starting from a given composition, it iteratively breaks down likely
    synthesis paths up to a certain number of elements, e.g., binaries or
    elements.
    """

    def __init__(self, entries, max_nelement=2):
        """

        :param entries ([ComputedEntry]): The computed entries from which to
            perform the phase diagram analysis.
        :param nelement_reactant (float): This sets a limit as to how many
            elements each reactant should have before the analysis is stopped.
            Default is 2 for binaries. Set to 1 for elements.
        """
        self.pd = PhaseDiagram(entries)
        self.stable_entries = self.pd.stable_entries
        self.max_nlement = max_nelement

    def get_reaction_tree(self, target):
        # Recursive algo to get all reactions
        target = Composition(target)

        def _get_tree(parent, to_remove):
            to_remove = set(to_remove)
            to_remove.add(target.reduced_formula)
            new_stable = [e for e in self.stable_entries if
                          e.composition.reduced_formula not in to_remove]
            pd = PhaseDiagram(new_stable)
            decomp = pd.get_decomposition(target)
            rx_str = " + ".join(
                sorted([e.composition.reduced_formula for e in decomp.keys()]))
            child = Node(rx_str, parent, decomp=decomp,
                         avg_nelements=np.mean([len(e.composition) for e in decomp.keys()]))
            for e in decomp.keys():
                if not len(e.composition) <= self.max_nlement:
                    to_remove.add(e.composition.reduced_formula)
                    _get_tree(child, to_remove)
            return parent

        t = Node(target.reduced_formula,
                 decomp={PDEntry(target, 0): 1},
                 avg_nelements=len(target))

        return _get_tree(t, set())


def print_rxn_tree(rxn_tree, target, balanced_rxn_str=False):
    for pre, fill, node in RenderTree(rxn_tree):
        if balanced_rxn_str:
            rxn = Reaction(sorted([e.composition for e in node.decomp.keys()]),
                           [Composition(target)])
            name = str(rxn).split("-")[0]
        else:
            name = node.name
        print("%s%s (avg_nelements = %.2f)" % (pre, name, node.avg_nelements))


from pymatgen.util.testing import PymatgenTest


class PDSynthesisTreeTest(PymatgenTest):

    @classmethod
    def setUpClass(cls):
        mpr = MPRester()
        cls.lfo_entries = mpr.get_entries_in_chemsys(["Li", "Fe", "O"])

    def test_get_reactions(self):
        target = "LiFeO2"
        a = PDSynthesisTree(self.lfo_entries)
        rxn_tree = a.get_reaction_tree(target)
        print_rxn_tree(rxn_tree, target)
        # Balancing rxn is costly due to inefficient implementation for now,
        # and creates lots of visual noise. We do this for this small system to
        # illustrate what it is for. In most cases, we just want to know the phases participating in the reaction.
        print_rxn_tree(rxn_tree, target, balanced_rxn_str=True)


if __name__ == "__main__":
    import unittest
    unittest.main()