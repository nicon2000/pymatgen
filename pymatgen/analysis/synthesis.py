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


def is_element_or_binary(entry):
    return len(entry.composition) <= 2


__author__ = "Alex Dunn, Anubhav Jain, Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"



class SynthesisPathAnalyzer:

    def __init__(self, entries):
        self.pd = PhaseDiagram(entries)
        self.stable_entries = self.pd.stable_entries

    def get_reaction_tree(self, target, to_remove=None, parent=None):
        # Recursive algo to get all reactions
        target = Composition(target)
        if to_remove is None:
            to_remove = {}
        if parent is None:
            parent = Node(target.reduced_formula,
                          decomp={PDEntry(target, 0): 1})
        to_remove = set(to_remove)
        to_remove.add(target.reduced_formula)
        new_stable = [e for e in self.stable_entries if
                      e.composition.reduced_formula not in to_remove]
        pd = PhaseDiagram(new_stable)
        decomp = pd.get_decomposition(target)
        rx_str = " + ".join(
            sorted([e.composition.reduced_formula for e in decomp.keys()]))
        child = Node(rx_str, parent, decomp=decomp)
        for e in decomp.keys():
            if not is_element_or_binary(e):
                to_remove.add(e.composition.reduced_formula)
                self.get_reaction_tree(target, to_remove, parent=child)
        return parent


def print_rxn_tree(rxn_tree, target, balanced_rxn_str=False):
    for pre, fill, node in RenderTree(rxn_tree):
        if balanced_rxn_str:
            rxn = Reaction(sorted([e.composition for e in node.decomp.keys()]),
                           [Composition(target)])
            name = str(rxn).split("-")[0]
        else:
            name = node.name
        print("%s%s" % (pre, name))


from pymatgen.util.testing import PymatgenTest

class SynthesisPathAnalyzerTest(PymatgenTest):

    @classmethod
    def setUpClass(cls):
        mpr = MPRester()
        cls.lfo_entries = mpr.get_entries_in_chemsys(["Li", "Fe", "O"])

    def test_get_reactions(self):
        target = "LiFeO2"
        a = SynthesisPathAnalyzer(self.lfo_entries)
        rxn_tree = a.get_reaction_tree(target)
        print_rxn_tree(rxn_tree, target)
        # Balancing rxn is costly due to inefficient implementation for now, and creates lots of
        # visual noise. We do this for this small system to illustrate what it is for.
        # In most cases, we just want to know the phases participating in the reaction.
        print_rxn_tree(rxn_tree, target, balanced_rxn_str=True)

if __name__ == "__main__":
    import unittest
    unittest.main()