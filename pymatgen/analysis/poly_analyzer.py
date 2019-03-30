from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import get_el_sp
from enum import Enum
import itertools


class PolySharing(Enum):
    FACE = 3
    EDGE = 2
    CORNER = 1
    NON = 0


class CoordPoly:
    """
    Helper class
    """

    def __init__(self, center, anions):
        self.center = center
        self.anion_sites = []
        self.anion_indices = []
        for s, d, i in anions:
            self.anion_sites.append(s)
            self.anion_indices.append(i)

    def get_sharing(self, coord_poly):
        shared = set(self.anion_sites).intersection(coord_poly.anion_sites)
        return PolySharing(len(shared))

    def __str__(self):
        comp = self.center.species
        for s in self.anion_sites:
            comp += s.species
        return "%s centered at %s" % (comp.reduced_formula, self.center.frac_coords)


class PolyStructure(Structure):
    """
    Breakdown of structure into polyhedra. Only cutoff based NN finding
    implemented for now. Can be extended to other NN algos in future.
    """


    def __init__(self, structure: Structure, anion_species: list = None,
                 cutoff_radius: float = None):
        """

        :param structure: Structure
        :param anion_species: List of anion species. Defaults to most
            electronegative species in structure.
        :param cutoff_radius: A cutoff radius. Defaults to sum of radii * 1.1.
        """
        if anion_species is None:
            anion_species = {sorted(structure.composition.keys())[-1]}
        else:
            anion_species = set([get_el_sp(sp) for sp in anion_species])
        anion_radius = sum([k.atomic_radius for k in anion_species])
        polys = []
        for site in structure:
            if not anion_species.intersection(site.species):
                if cutoff_radius is None:
                    r = 1.5 * (sum([k.atomic_radius for k in site.species.keys()]) +
                               anion_radius)
                else:
                    r = cutoff_radius
                nn = structure.get_neighbors(site, r=r, include_index=True)
                nn = [l for l in nn if anion_species.intersection(l[0].species)]
                polys.append(CoordPoly(site, nn))

        self.structure = structure
        self.polys = polys
        self.anion_species = anion_species
        self.cutoff_radius = cutoff_radius


from pymatgen.util.testing import PymatgenTest


class PolyStructureTest(PymatgenTest):

    def test_init(self):
        structure = self.get_structure("LiFePO4")
        polystructure = PolyStructure(structure)
        for poly1, poly2 in itertools.combinations(polystructure.polys, 2):
            sharing = poly1.get_sharing(poly2)
            if sharing != PolySharing.NON:
                print(poly1)
                print(poly2)
                print(sharing)
                print()


if __name__ == "__main__":
    import unittest
    unittest.main()
