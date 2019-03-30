from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.core.periodic_table import get_el_sp
from enum import Enum
import itertools
import numpy as np


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
        self.anions = anions
        self.anion_sites = []
        self.anion_indices = []
        comp = center.species
        for s, d, i in anions:
            self.anion_sites.append(s)
            self.anion_indices.append(i)
            comp += s.species
        self.composition = comp

    def get_sharing(self, coord_poly):
        mappings = []
        for s1, s2 in itertools.product(self.anion_sites, coord_poly.anion_sites):
            if np.allclose(s1.coords, s2.coords):
                mappings.append([s1, s2])
        return PolySharing(len(mappings))

    def __str__(self):
        return "%s %s" % (self.composition.reduced_formula,
                          self.center.frac_coords)


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

    def print_poly_sharing(self):
        # Find self-sharing with itself across PBC.
        for poly in self.polys:
            for image in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
                if not np.allclose(image, [0, 0, 0]):
                    c = poly.center
                    c = PeriodicSite(c.species, c.frac_coords + image, c.lattice)
                    asites = poly.anions
                    asites = [(PeriodicSite(s.species, s.frac_coords + image, s.lattice), d, i)
                              for s, d , i in asites]
                    image_poly = CoordPoly(c, asites)
                    sharing = poly.get_sharing(image_poly)
                    if sharing != PolySharing.NON:
                        print("Poly %s shares %s with itself" % (poly, sharing.name))
                        break

        for poly1, poly2 in itertools.combinations(self.polys, 2):
            sharing = poly1.get_sharing(poly2)
            if sharing != PolySharing.NON:
                print("%s-%s: %s" % (poly1.composition.reduced_formula,
                                     poly2.composition.reduced_formula,
                                     sharing.name))


from pymatgen.util.testing import PymatgenTest


class PolyStructureTest(PymatgenTest):

    def test_init(self):
        structure = self.get_structure("Li2O")
        polystructure = PolyStructure(structure)
        polystructure.print_poly_sharing()


if __name__ == "__main__":
    import unittest
    unittest.main()
