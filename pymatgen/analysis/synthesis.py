# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals

import logging

from pymatgen import MPRester, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
from pymatgen.analysis.reaction_calculator import Reaction, ComputedReaction
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

    def __init__(self, entries, target, max_nelements=2):
        """
        Args:
            entries ([ComputedEntry]): The computed entries from which to
                perform the phase diagram analysis.
            target (Composition/str): Target composition to get synthesis tree
                for.
            max_nelements (float): This sets a limit as to how many
                elements each reactant should have before the analysis is
                stopped. Default is 2 for binaries. Set to 1 for elements.
        """
        self.pd = PhaseDiagram(entries)
        self.stable_entries = self.pd.stable_entries
        self.max_nelements = max_nelements
        self.target = PDEntry(target, 0)

        def _get_tree(parent, to_remove):
            # Recursive algo to get all reactions
            to_remove = set(to_remove)
            to_remove.add(self.target.composition.reduced_formula)
            new_stable = [e for e in self.stable_entries if
                          e.composition.reduced_formula not in to_remove]
            pd = PhaseDiagram(new_stable)
            decomp = pd.get_decomposition(self.target.composition)
            rx_str = " + ".join(
                sorted([e.composition.reduced_formula for e in decomp.keys()]))
            child = Node(rx_str, parent, decomp=decomp,
                         avg_nelements=np.mean([len(e.composition)
                                                for e in decomp.keys()]))
            for e in decomp.keys():
                if not len(e.composition) <= self.max_nelements:
                    to_remove.add(e.composition.reduced_formula)
                    _get_tree(child, to_remove)
            return parent

        t = Node(self.target.composition.reduced_formula,
                 decomp={self.target: 1},
                 avg_nelements=len(target))

        self.rxn_tree = _get_tree(t, set())

    def get_unique_reactions(self):
        nodes = []
        names = set()
        for pre, fill, node in RenderTree(self.rxn_tree):
            if node.name not in names:
                nodes.append(node)
                names.add(node.name)

        return sorted(nodes, key=lambda n: n.avg_nelements)

    @classmethod
    def from_mp(cls, chemsys, **kwargs):
        mpr = MPRester()
        entries = mpr.get_entries_in_chemsys(chemsys)
        return PDSynthesisTree(entries, **kwargs)

    def print(self, balanced_rxn_str=False):
        for pre, fill, node in RenderTree(self.rxn_tree):
            if balanced_rxn_str:
                rxn = ComputedReaction(sorted([e for e in node.decomp.keys()],
                                              key=lambda e: e.composition.reduced_formula),
                                       [self.target])

                name = str(rxn).split("-")[0]
                output = "%s%s (avg_nelements = %.2f, energy = %.3f)" % (
                    pre, name, node.avg_nelements, rxn.calculated_reaction_energy)
            else:
                name = node.name
                output = "%s%s (avg_nelements = %.2f)" % (pre, name,
                                                          node.avg_nelements)
            print(output)


from pymatgen.util.testing import PymatgenTest


class PDSynthesisTreeTest(PymatgenTest):

    @classmethod
    def setUpClass(cls):
        from monty.serialization import loadfn
        import pymatgen
        import os
        test_path = os.path.join(
            os.path.abspath(os.path.dirname(pymatgen.__file__)), "..",
            "test_files")
        cls.lfo_entries = loadfn(os.path.join(test_path, "Li-Fe-O.json"))

    def test_func(self):
        target = "LiFeO2"
        rxn_tree = PDSynthesisTree(self.lfo_entries, target)
        rxn_tree.print()
        # Balancing rxn is costly due to inefficient implementation for now,
        # and creates lots of visual noise. We do this for this small system to
        # illustrate what it is for. In most cases, we just want to know the
        # phases participating in the reaction.
        rxn_tree.print(balanced_rxn_str=True)

        # This breaks everything to elements
        a = PDSynthesisTree(self.lfo_entries, target, 1)
        rxn_tree.print()

        for rxn in a.get_unique_reactions():
            print("%s (avg_nelements = %.2f)" % (rxn.name, rxn.avg_nelements))


if __name__ == "__main__":
    import unittest
    unittest.main()
