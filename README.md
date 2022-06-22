# Lcap

Lcap is a small library for active learning of the intersection of several regular languages, on top of active automata-learning libraries.
The initial version of Lcap is based on:

- Sebastian Junges and Jurriaan Rot, Learning Language Intersections, 2022

Currently, Lcap is build on top of two libraries:

- dfa
- aalpy

## Installing Lcap

Run 

```pip install dfa aalpy click```


## Running experiments

The script ```lcap-experiments.py``` allows you to rerun the experiments of [1].
The script produces some outputs on the command line, but more importantly, it creates two files  corresponding with the two tables in [1]:
- ```table1.tex``` and 
- ```table2.tex```

These files contain tables in latex that can be compiled into pdf.


To run all benchmarks once, run 
```
python lcap-experiment.py -b mod -b imod -l indep -l wbw -l mbm
```

As the learning algorithms are randomized, experiments should be run multiple times:
```
python lcap-experiment.py -n 20 -b mod -b imod -l indep -l wbw -l mbm
```

You can also change the seed `--seed X` or select only a subset of learners (by removing some `-l` options) and benchmarks (by removing some `-b` options).

## Using Lcap

The algorithms are implemented in `lcap/learners.py`. The `lcap-experiments.py` script exemplifies the usage. Note that we currently require the membership/equivalence oracles to be based on the actual automaton (ground truth).

## Reference results

In the folder `results/` you can find reference results for published experiments.


