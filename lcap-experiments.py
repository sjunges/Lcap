from statistics import mean
from dfa import DFA
import lcap.learners

import click
import random
import dfa.utils


class learner:
    def __init__(self, Strategy, name, learner_uses_cache):
        self.Strategy = Strategy
        self.learner_uses_cache = learner_uses_cache
        self.name = name


LEARNERS = {"indep": learner(lcap.learners.IndependentLearner, "indep", True),
            "wbw": learner(lcap.learners.LazyWordByWordLearner, "wbw", False),
            "mbm": learner(lcap.learners.MachineByMachineLearner, "mbm", False)}

class Benchmark:
    """
    Struct for holding benchmark info
    """
    def __init__(self, identifier, input):
        self.identifier = identifier
        self.input = input


class BenchmarkRunner:
    """
    Class to run benchmarks and output statistics.
    """
    def __init__(self, filename, repetitions, learners):
        self._repetitions = repetitions
        self._filename = filename
        self._learners = learners
        with open(filename, "w") as f:
            f.write("\\documentclass{article}\n\\usepackage{graphics}\n\\usepackage{multicol}\n\\usepackage{amsmath}\n\\begin{document}")
            f.write("% Table generated by lcap experiments script\n ")
            f.write("\\begin{table}[t]\n\\centering\n\\scalebox{0.85}{\n\\begin{tabular}[t]{l|rrrr|rrrr|rrrr} & \\multicolumn{4}{c|}{Independent} & \\multicolumn{4}{c|}{Word-By-Word} & \\multicolumn{4}{c}{Machine-By-Machine} \\\\\n")
            f.write(" & $\\sharp$MQ & SC$_{\\text{MQ}}$ & $\\sharp$EQ & SC$_{\\text{EQ}}$   & $\\sharp$MQ & SC$_{\\text{MQ}}$ & $\\sharp$EQ & SC$_{\\text{EQ}}$   & $\\sharp$MQ & SC$_{\\text{MQ}}$ & $\\sharp$EQ & SC$_{\\text{EQ}}$    \\\\\\hline\n")
    def _check_equal(self, auts, result):
        res = auts[0]; [res := res & x for x in auts[1:]]
        lhs = dfa.utils.minimize(res)
        symdiff = dfa.utils.minimize(lhs ^ result)
        return len(symdiff.states()) == 1 and symdiff.label("") == 0

    def set_benchmark_id(self, identifier):
        with open(self._filename, 'a') as f:
            f.write(identifier)

    def close(self):
        with open(self._filename, 'a') as f:
            f.write("\\end{tabular}}\n\end{table}\n\\end{document}")

    def benchmark_list(self, benchmarks):
        for nr, tc in enumerate(benchmarks):
            assert isinstance(tc,Benchmark)
            self.set_benchmark_id(identifier=tc.identifier)
            print(f"Running {tc.identifier} ({nr+1}/{len(benchmarks)})")
            self.run_test_case(tc.input)

    def run_test_case(self, test_case):
        with open(self._filename, 'a') as f:
            for learner_def in self._learners:
                print(f"\t Use {learner_def.name}")
                learner = learner_def.Strategy(test_case,
                                               learner_uses_cache=learner_def.learner_uses_cache,
                                               cache_sul_queries=True)
                queries_learning = []
                steps_learning = []
                queries_eq = []
                steps_eq = []
                incorrect = False
                for sv in range(self._repetitions):
                    print(f"\t\tRun {sv+1}/{self._repetitions}")
                    # Ensure that individual runs are reproducible
                    random.seed(sv)
                    learner.reset()
                    learned_intersection_ind = learner.run()
                    correct = self._check_equal(test_case, learned_intersection_ind)
                    if not correct:
                        print("\t\t\tIncorrect result!")
                        incorrect = True
                        break
                    print(f"\t\t\t{learner.stats}")
                    queries_learning.append(learner.stats["queries_learning"])
                    steps_learning.append(learner.stats["steps_learning"])
                    queries_eq.append(learner.stats["learning_rounds"])
                    steps_eq.append(learner.stats["sul_steps"])
                if incorrect:
                    f.write("\t& ")
                    f.write("INC")
                    f.write("\t& ")
                    f.write("INC")
                    f.write("\t& ")
                    f.write("INC")
                    f.write("\t& ")
                    f.write("INC")
                else:
                    f.write("\t& ")
                    f.write(str(int(mean(queries_learning))))
                    f.write("\t& ")
                    f.write(str(int(mean(steps_learning))))
                    f.write("\t& ")
                    f.write(str(int(mean(queries_eq))))
                    f.write("\t& ")
                    f.write(str(int(mean(steps_eq))))
            f.write("\\\\")
            f.write("\n")




def _create_mod_benchmarks():
    """
    Creates the table 1 in the paper "Learning Language Intersections"
    :param filename: The name of the latex file to write the results to
    """
    def _make_mod_dfa(modulo_value):
        return DFA(
            start=0,
            inputs={0, 1},
            label=lambda s: s == 0,
            transition=lambda s, c: (s + c) % modulo_value
        )

    def _create_mod_languages_benchmark(modulo_values):
        identifier = "MOD(" + ",".join([str(x) for x in modulo_values]) + ")"
        test_case = [_make_mod_dfa(mv) for mv in modulo_values]
        return Benchmark(identifier, test_case)

    descriptions = [[30], [2, 3, 5], [2, 3, 5, 6, 10], [2, 3, 5, 30],
                   [2, 3, 5, 6, 10, 15, 30], list(reversed([2, 3, 5, 6, 10, 15, 30])),
                   list(reversed([2, 3, 5, 6, 10, 15])), [24], [3, 8], [3, 8, 24], [2, 3, 8],
                   [2, 3, 6, 8, 12, 24], list(reversed([2, 3, 6, 8, 12, 24])),
                   list(reversed([3, 6, 2, 8, 12])), [32], [32, 16, 8],
                   [8, 16, 32], [32, 16, 8, 4, 2], list(reversed([32, 16, 8, 4, 2]))]
    return [_create_mod_languages_benchmark(description) for description in descriptions]


def _create_imod_benchmarks():
    def _make_imod_dfa(modulo_value):
        return DFA(
            start=0,
            inputs={0, 1},
            label=lambda s: s != 0,
            transition=lambda s, c: (s + c) % modulo_value
        )

    def _create_imod_languages(modulo_values):
        identifier = "IMOD(" + ",".join([str(x) for x in modulo_values]) + ")"
        test_case = [_make_imod_dfa(mv) for mv in modulo_values]
        return Benchmark(identifier, test_case)

    descriptions = [[2], [30], [2, 3, 5], [2, 3, 5, 6, 10], [2,3,5,6,10,15,30],
                    list(reversed([2, 3, 5, 6, 10, 15, 30])), list(reversed([2, 3, 5, 6, 10, 15])),
                    [3, 8], [3, 6, 8, 12, 18], [3, 6, 8, 12, 16, 18, 21], [8], [16, 8],
                    [8, 16], [32, 24, 16, 8], [8, 16, 24, 32], [8, 4, 2], list(reversed([8, 4, 2])),
                    [32, 16, 8, 4, 2], list(reversed([32, 16, 8, 4, 2])), [2, 10, 20], [20, 10, 2]]
    return [_create_imod_languages(description) for description in descriptions]


@click.command()
@click.option('--seed', default=0, help='Random seed')
@click.option('--repetitions', '-n', default=1, help="How often to repeat each benchmark")
@click.option('--benchmark', '-b', multiple=True, type=click.Choice(["mod", "imod"]), required=True)
@click.option('--learner', '-l', multiple=True, type=click.Choice(["indep", "wbw", "mbm"]), required=True)
def main(seed, repetitions, benchmark, learner):
    random.seed(seed)
    use_these_learners = [LEARNERS[l] for l in learner]
    for b in benchmark:
        if b == "mod":
            runner = BenchmarkRunner("table1.tex", repetitions, use_these_learners)
            runner.benchmark_list(_create_mod_benchmarks())
            runner.close()
        if b == "imod":
            runner = BenchmarkRunner("table2.tex", repetitions, use_these_learners)
            runner.benchmark_list(_create_imod_benchmarks())
            runner.close()


if __name__ == '__main__':
    main()


