from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import get_el_sp
from enum import Enum
import itertools
import collections
import numpy as np


class PolySharing(Enum):
    SFACE = 4  # Face-sharing cubes, e.g., CsCl.
    TFACE = 3  # Typical triangle face-sharing of tets and octs.
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
                # We compare cartesian coordinates.
                mappings.append([s1, s2])
        return PolySharing(len(mappings))

    def __repr__(self):
        output = ["Center: %s" % (self.center)]
        for a in self.anion_sites:
            output.append("Anion: %s" % (a))
        return "\n".join(output)

    def __str__(self):
        return "%s %s" % (self.composition.reduced_formula,
                          self.center.frac_coords)


class PolyStructure(Structure):
    """
    Breakdown of structure into polyhedra. Only cutoff based NN finding
    implemented for now. Can be extended to other NN algos in future.
    """

    def __init__(self, structure: Structure, anion_species: list = None,
                 cutoff_radius: float = None, symprec: float =  0.1):
        """

        Args:
            structure: Structure
            anion_species: List of anion species. Defaults to most
                electronegative species in structure.
            cutoff_radius: A cutoff radius. Defaults to sum of radii * 1.2.
            symprec: Symmetry precision for determining spacegroup and
                equivalent sites.
        """
        a = SpacegroupAnalyzer(structure, symprec=symprec)
        symmetrized_structure = a.get_symmetrized_structure()

        if anion_species is None:
            anion_species = {sorted(structure.composition.keys())[-1]}
        else:
            anion_species = set([get_el_sp(sp) for sp in anion_species])

        def is_anion_site(site):
            return anion_species.intersection(site.species)

        # We will use the largest anion for anion radius.
        anion_radius = max([k.atomic_radius for k in anion_species])

        polys = []
        for sites in symmetrized_structure.equivalent_sites:
            if cutoff_radius is None:
                # 1.2 * (Weighted sum of cation radius + anion radius)
                r = 1.2 * (sum([k.atomic_radius * v for k, v in sites[0].species.items()]) +
                           anion_radius)
            else:
                r = cutoff_radius
            if not is_anion_site(sites[0]):
                poly_site = []
                for site in sites:
                    nn = structure.get_neighbors(site, r=r, include_index=True)
                    nn = [l for l in nn if is_anion_site(l[0])]
                    poly_site.append(CoordPoly(site, nn))
                polys.append(poly_site)

        self.structure = structure
        self.symmetrized_structure = symmetrized_structure
        self.polys = polys
        self.anion_species = anion_species
        self.cutoff_radius = cutoff_radius

    def get_connections(self):
        connections = collections.defaultdict(set)
        n = len(self.polys)
        for i1 in range(n):
            for i2 in range(i1, n):
                connect_index = tuple(sorted([i1, i2]))
                poly1 = self.polys[i1][0]
                fcoords1 = poly1.center.frac_coords
                for poly2 in self.polys[i2]:
                    c = poly2.center
                    for image in itertools.product([0, -1, 1], [0, -1, 1],
                                                   [0, -1, 1]):
                        fcoords2 =  c.frac_coords + image
                        if not np.allclose(fcoords2, fcoords1):
                            c = PeriodicSite(c.species, fcoords2, c.lattice)
                            asites = [(PeriodicSite(s.species,
                                                    s.frac_coords + image,
                                                    s.lattice), d, i)
                                      for s, d, i in poly2.anions]
                            image_poly = CoordPoly(c, asites)
                            sharing = poly1.get_sharing(image_poly)
                            if sharing != PolySharing.NON:
                                connections[connect_index].add(sharing)
                                break
        return connections

    def print_connections(self):
        connections = self.get_connections()
        for k, v in connections.items():
            i1, i2 = k
            print("%s-%s: %s" % (self.polys[i1][0].composition.reduced_formula,
                                 self.polys[i2][0].composition.reduced_formula,
                                 v))


from pymatgen.util.testing import PymatgenTest


class PolyStructureTest(PymatgenTest):

    def test_init(self):
        structure = self.get_structure("LiFePO4")
        polystructure = PolyStructure(structure)
        polystructure.print_connections()
        structure = self.get_structure("Li2O")
        polystructure = PolyStructure(structure)
        polystructure.print_connections()
        structure = self.get_structure("CsCl")
        polystructure = PolyStructure(structure)
        polystructure.print_connections()
        structure = self.get_structure("Li10GeP2S12")
        polystructure = PolyStructure(structure)
        polystructure.print_connections()

if __name__ == "__main__":
    import unittest
    unittest.main()
