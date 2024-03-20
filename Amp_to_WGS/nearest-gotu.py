import click
import biom
import qiime2
import q2_types
import bp
import re
import skbio
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import numpy as np
from functools import reduce
from operator import or_


def is_gotu(name):
    GOTU = re.compile(r'^G\d{9}$')
    return GOTU.match(name) is not None


def test_is_gotu():
    tests = [('G123456789', True),
             ('asdasd', False),
             ('G123', False),
             ('AAAAAAAAAA', False),
             ('1111111111111', False),
             ('111111111', False)]
    for t, exp in tests:
        obs = is_gotu(t)
        assert obs == exp


def index_gotus(tree):
    gotus = sorted([n.name for n in tree.tips() if is_gotu(n.name)])
    gotu_index = {i: idx for idx, i in enumerate(gotus)}
    gotu_index_inv= {idx: i for i, idx in gotu_index.items()}
    number_of_gotus = len(gotu_index)

    assert number_of_gotus > 0

    for n in tree.postorder(include_self=True):
        n.gotu_distance = np.zeros(number_of_gotus)
        if n.is_tip():
            if is_gotu(n.name):
                n.gotus = np.array([gotu_index[n.name], ], dtype=int)
            else:
                n.gotus = np.array([], dtype=int)

        if not n.is_tip():
            n.gotus = np.hstack([c.gotus for c in n.children])

            for c in n.children:
                if c.gotus.size > 0:
                    tmp = c.gotu_distance.copy()
                    tmp[c.gotus] += c.length
                    n.gotu_distance += tmp

    return gotu_index, gotu_index_inv


def test_index_gotus():
    t = skbio.TreeNode.read(["((G123456789:0.1,A:0.2)i1:0.3,(B:0.3,G900000000:0.4)i2:0.5)root;"])
    idx, idx_inv = index_gotus(t)
    npt.assert_almost_equal(t.gotu_distance, np.array([0.4, 0.9]))
    npt.assert_almost_equal(t.find('i1').gotu_distance, np.array([0.1, 0.0]))
    npt.assert_almost_equal(t.find('i2').gotu_distance, np.array([0.0, 0.4]))
    npt.assert_almost_equal(t.find('G123456789').gotu_distance, np.array([0.0, 0.0]))
    npt.assert_almost_equal(t.find('G900000000').gotu_distance, np.array([0.0, 0.0]))
    npt.assert_almost_equal(t.find('A').gotu_distance, np.array([0.0, 0.0]))
    npt.assert_almost_equal(t.find('B').gotu_distance, np.array([0.0, 0.0]))
    assert idx == {'G123456789': 0, 'G900000000': 1}
    assert idx_inv == {0: 'G123456789', 1: 'G900000000'}


def search_asvs(tree, idx_inv):
    asvs = sorted([n.name for n in tree.tips() if not is_gotu(n.name)])

    results = []
    for asv in asvs:
        tip = tree.find(asv)
        distance = tip.length
        for anc in tip.ancestors():
            if anc.gotu_distance.any():
                # noice
                # https://stackoverflow.com/a/7164681/19741
                masked = np.ma.masked_equal(anc.gotu_distance, 0.0, copy=False)
                distance += masked.min()
                index = masked.argmin()
                gotu = idx_inv[index]
                results.append([asv, gotu, distance])
                break
            else:
                distance += anc.length

    return pd.DataFrame(results,
                        columns=['FeatureID', 'gOTU', 'distance']).set_index('FeatureID')


def test_search_asvs():
    t = skbio.TreeNode.read(["((G123456789:0.1,A:0.2)i1:0.3,(B:0.3,G900000000:0.4)i2:0.5)root;"])
    exp = pd.DataFrame([['A', 'G123456789', 0.3],
                        ['B', 'G900000000', 0.7]],
                       columns=['FeatureID', 'gOTU', 'distance']).set_index('FeatureID')
    idx, idx_inv = index_gotus(t)
    obs = search_asvs(t, idx_inv)
    pdt.assert_frame_equal(obs, exp)

    # make sure we resolve multiple possibilties
    t = skbio.TreeNode.read(["(((G123456789:0.1,G987654321:0.2)i1:0.3,(A:0.1,B:0.2)i2:0.4)i3:0.5,(C:0.3,G900000000:0.4)i4:0.5)root;"])
    exp = pd.DataFrame([['A', 'G123456789', 0.9],
                        ['B', 'G123456789', 1.0],
                        ['C', 'G900000000', 0.7]],
                       columns=['FeatureID', 'gOTU', 'distance']).set_index('FeatureID')
    idx, idx_inv = index_gotus(t)
    obs = search_asvs(t, idx_inv)
    pdt.assert_frame_equal(obs, exp)


def tests():
    test_is_gotu()
    test_index_gotus()
    test_search_asvs()


@click.command()
@click.option('--feature-table', type=click.Path(exists=True), required=True)
@click.option('--phylogeny', type=click.Path(exists=True), required=True)
@click.option('--output', type=click.Path(exists=False), required=True)
def nearest(feature_table, phylogeny, output):
    #raise ValueError()
    if feature_table.endswith('.qza'):
        feature_table = qiime2.Artifact.load(feature_table).view(biom.Table)
        phylogeny = qiime2.Artifact.load(phylogeny).view(q2_types.tree.NewickFormat)
    else:
        feature_table = biom.load_table(feature_table)
    with open(str(phylogeny)) as treefile:
        # read balanced parentheses tree
        phylogeny = bp.parse_newick(treefile.readline())
        phylogeny = phylogeny.shear(set(feature_table.ids(axis='observation'))).collapse()
        phylogeny = bp.to_skbio_treenode(phylogeny)

    idx, idx_inv = index_gotus(phylogeny)
    matches = search_asvs(phylogeny, idx_inv)
    matches.to_csv(output, sep='\t', index=True, header=False)


if __name__ == '__main__':
    tests()
    nearest()
