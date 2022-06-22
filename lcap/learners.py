from collections import Iterable
import dfa.utils
import lstar
import aalpy
import aalpy.utils
import aalpy.SULs
import aalpy.oracles
import aalpy.learning_algs


def make_universal_automaton(alphabet):
    return dfa.DFA(start=0,
                   inputs=alphabet,
                   label=lambda s: True,
                   transition=lambda s, c: s)


class CacheSUL(aalpy.base.SUL):
    """
    System under learning that keeps a multiset of all queries in memory.
    This multiset/cache is encoded as a tree.
    """

    def __init__(self, sul: aalpy.base.SUL):
        super().__init__()
        self.sul = sul
        self.cache = aalpy.base.CacheTree.CacheTree()

    def query(self, word):
        """
        Performs a membership query on the SUL if and only if `word` is not a prefix of any trace in the cache.
        Before the query, pre() method is called and after the query post()
        method is called. Each letter in the word (input in the input sequence) is executed using the step method.

        Args:

            word: membership query (word consisting of letters/inputs)

        Returns:

            list of outputs, where the i-th output corresponds to the output of the system after the i-th input

        """
        cached_query = self.cache.in_cache(word)
        if cached_query:
            self.num_cached_queries += 1
            return cached_query

        # get outputs using default query method
        out = self.sul.query(word)

        # add input/outputs to tree
        self.cache.reset()
        for i, o in zip(word, out):
            self.cache.step_in_cache(i, o)

        self.num_queries += 1
        self.num_steps += len(word)
        return out

    def pre(self):
        """
        Reset the system under learning and current node in the cache tree.
        """
        self.cache.reset()
        self.sul.pre()

    def post(self):
        self.sul.post()

    def step(self, letter):
        """
        Executes an action on the system under learning, adds it to the cache and returns its result.

        Args:

           letter: Single input that is executed on the SUL.

        Returns:

           Output received after executing the input.

        """
        out = self.sul.step(letter)
        self.cache.step_in_cache(letter, out)
        return out


def learn_automaton_with_lstar(secret_automaton, filter_automaton):
    return lstar.learn_dfa(secret_automaton.inputs, secret_automaton.label,
                           lstar.iterative_deeping_ce(secret_automaton.label, depth=10))


def _create_aalpy_dfa_from_automaton(automaton):
    transitions_and_initial = dfa.utils.dfa2dict(automaton)
    return aalpy.utils.dfa_from_state_setup(transitions_and_initial[0])


def _aalpy2dict(aalpy_dfa):
    result = {}
    for state in aalpy_dfa.states:
        transitions = dict()
        for i, dest in state.transitions.items():
            transitions[i] = dest.state_id
        result[state.state_id] = (state.is_accepting, transitions)
    return result


def _create_dfa_from_aalpy_automaton(automaton):
    return dfa.utils.minimize(dfa.utils.dict2dfa(_aalpy2dict(automaton), "s0"))


def _to_aalpy_trace(trace):
    return trace


class _FilterDfaSUL(aalpy.base.SUL):
    def __init__(self, dfa, filter, use_up_to_memberships_queries, cache_sul):
        self._system_under_learning = aalpy.SULs.DfaSUL(dfa)
        if cache_sul:
            self._system_under_learning = CacheSUL(self._system_under_learning)
        ## We use the SUL to have a query interface
        self._filter_sul = aalpy.SULs.DfaSUL(filter)
        self._use_up_to_membership_queries = use_up_to_memberships_queries
        self._num_eq_steps = 0

    def query(self, word: tuple) -> list:

        filter_res = self._filter_sul.query(word)
        if len(word) > 0 and self._use_up_to_membership_queries:
            # TODO we can avoid searching the complete word by reversing.

            positives = [i for i, val in enumerate(filter_res) if val]
            if len(positives) == 0:
                return filter_res
            last_positive = positives[-1]
            dfa_res = self._system_under_learning.query(word[:last_positive + 1])
            res = [x & y for x, y in zip(filter_res[:last_positive + 1], dfa_res)] + [False] * (
                        len(word) - last_positive - 1)
            assert len(res) == len(word), f"Expected {len(res)} == {len(word)}"
            return res
        else:
            if filter_res[-1]:
                res = self._system_under_learning.query(word)
            else:
                res = filter_res
            return res

    def pre(self):
        self._cached_steps = []
        self._system_under_learning.pre()
        self._filter_sul.pre()

    def post(self):
        self._system_under_learning.post()
        self._filter_sul.post()

    def step(self, token):
        filter_step_res = self._filter_sul.step(token)
        self._cached_steps += [token]
        if filter_step_res:
            for t in self._cached_steps:
                self._num_eq_steps += 1
                sul_step_res = self._system_under_learning.step(t)
            self._cached_steps = []
            return sul_step_res
        else:
            return False

    @property
    def num_queries(self):
        return self._system_under_learning.num_queries

    @property
    def num_steps(self):
        return self._system_under_learning.num_steps

    @property
    def num_eq_steps(self):
        return self._num_eq_steps


class _EagerIntersectionDfaSUL(aalpy.base.SUL):
    def __init__(self, auts):
        self._suls = [aalpy.SULs.DfaSUL(aut) for aut in auts]
        self._num_eq_steps = 0

    def pre(self):
        for sul in self._suls:
            sul.pre()

    def post(self):
        for sul in self._suls:
            sul.post()

    def step(self, token):

        res = True
        for sul in self._suls:
            self._num_eq_steps += 1
            res &= sul.step(token)
        return res

    def query(self, word: tuple) -> list:
        res = [True] * max(len(word), 1)
        for sul in self._suls:
            new_res = sul.query(word)
            res = [x & y for x, y in zip(res, new_res)]
        return res

    @property
    def num_queries(self):
        return sum([sul.num_queries for sul in self._suls])

    @property
    def num_steps(self):
        return sum([sul.num_steps for sul in self._suls])

    @property
    def num_eq_steps(self):
        return self._num_eq_steps


class _IntersectionDfaSUL(aalpy.base.SUL):
    def __init__(self, auts, use_membership_up_to, cache_suls):
        self._suls = [aalpy.SULs.DfaSUL(aut) for aut in auts]
        if cache_suls:
            self._suls = [CacheSUL(sul) for sul in self._suls]
        self._use_membership_up_to = use_membership_up_to
        self._cached_steps = [[] for _ in range(len(self._suls))]
        assert len(self._cached_steps) == len(self._suls)
        self._num_eq_steps = 0

    def pre(self):
        self._cached_steps = [[] for _ in range(len(self._suls))]
        for sul in self._suls:
            sul.pre()

    def post(self):
        for sul in self._suls:
            sul.post()

    def step(self, token):
        for cache in self._cached_steps:
            cache += [token]
        res = True
        for index, sul in enumerate(self._suls):
            for t in self._cached_steps[index]:
                self._num_eq_steps += 1
                res = sul.step(t)
            self._cached_steps[index] = []
            if not res:
                return False
        return True

    def query(self, word: tuple) -> list:
        if len(word) > 0 and self._use_membership_up_to:
            last_positive = len(word)
            res = [True] * len(word)
            for sul in self._suls:
                new_res = sul.query(word[:last_positive + 1])
                positives = [i for i, val in enumerate(new_res) if val]
                if len(positives) == 0:
                    return [False] * len(word)
                res = [x & y for x, y in zip(res[:last_positive + 1], new_res)] + [False] * (
                        len(word) - last_positive - 1)
                last_positive = positives[-1]
                assert len(res) == len(word), f"Expected {len(res)} == {len(word)} "
            return res
        else:
            for sul in self._suls:
                res = sul.query(word)
                if not res[-1]:
                    return res
            return res

    @property
    def num_queries(self):
        return sum([sul.num_queries for sul in self._suls])

    @property
    def num_steps(self):
        return sum([sul.num_steps for sul in self._suls])

    @property
    def num_eq_steps(self):
        return self._num_eq_steps


class _FilteredEquivalenceOracle(aalpy.base.Oracle):
    def __init__(self, alphabet, filter_automaton, sul, oracle_type=aalpy.oracles.RandomWMethodEqOracle):
        self._filter_automaton = filter_automaton
        self._sul = sul
        self._oracle = oracle_type(alphabet, self._sul, walks_per_state=10, walk_len=100)

    def find_cex(self, hypothesis):
        hypothesis_dfa = _create_dfa_from_aalpy_automaton(hypothesis)
        cex = dfa.utils.find_subset_counterexample(hypothesis_dfa, self._filter_automaton)
        if cex is None:
            return self._oracle.find_cex(hypothesis)
        else:
            return _to_aalpy_trace(cex)

    @property
    def num_queries(self):
        return self._oracle.num_queries

    @property
    def num_steps(self):
        return self._oracle.num_steps


class DFALearner:
    def __init__(self, use_cache, cache_sul_queries):
        self._data = []
        self._library = "aalpy"
        self._use_cache = use_cache
        self._cache_sul_queries = cache_sul_queries

    def run(self, aut, filter=None, lazy=True):
        if self._library == "aalpy":
            dfa, data = learn_automaton_with_aalpy(aut, filter, use_cache=self._use_cache,
                                                   cache_sul=self._cache_sul_queries, lazy=lazy)
            self._data.append(data)
            return dfa
        elif self._library == "lstar":
            return learn_automaton_with_lstar(aut, filter)

    def reset(self):
        self._data = []

    @property
    def stats(self):
        return {
            "queries_learning": sum([data["queries_learning"] for data in self._data]),
            "queries_eq_oracle": sum([data["queries_eq_oracle"] for data in self._data]),
            "steps_learning": sum([data["steps_learning"] for data in self._data]),
            "learning_rounds": sum([data["learning_rounds"] for data in self._data]),
            "largest_automaton": max([data["automaton_size"] for data in self._data]),
            "sul_steps": sum([data["sul_steps"] for data in self._data])
        }


def learn_automaton_with_aalpy(secret_automaton, filter_automaton=None, use_cache=False, cache_sul=True, lazy=True):
    # TODO allow both filtering and having multiple secret_automata
    assert not (isinstance(secret_automaton, Iterable) and filter_automaton is not None)

    if isinstance(secret_automaton, Iterable):
        alphabet = list(secret_automaton[0].inputs)
        aalpy_dfas = [_create_aalpy_dfa_from_automaton(aut) for aut in secret_automaton]
    else:
        aalpy_dfa = _create_aalpy_dfa_from_automaton(secret_automaton)
        alphabet = list(secret_automaton.inputs)

    if filter_automaton is None:
        if isinstance(secret_automaton, Iterable):
            if lazy:
                sul_dfa = _IntersectionDfaSUL(aalpy_dfas, use_cache, cache_suls=cache_sul)
            else:
                sul_dfa = _EagerIntersectionDfaSUL(aalpy_dfas)
        else:
            sul_dfa = aalpy.SULs.DfaSUL(aalpy_dfa)
        oracle = aalpy.oracles.RandomWMethodEqOracle(alphabet, sul_dfa, walks_per_state=10, walk_len=100)
    else:
        aalpy_filter = _create_aalpy_dfa_from_automaton(filter_automaton)
        sul_dfa = _FilterDfaSUL(aalpy_dfa, aalpy_filter, use_cache, cache_sul=cache_sul)
        oracle = _FilteredEquivalenceOracle(alphabet, filter_automaton, sul_dfa)

    learned_dfa, data = aalpy.learning_algs.run_Lstar(alphabet, sul_dfa, oracle, automaton_type='dfa',
                                                      cache_and_non_det_check=use_cache, cex_processing='rs',
                                                      return_data=True, print_level=0)
    if hasattr(sul_dfa, "num_eq_steps"):
        data["sul_steps"] = sul_dfa.num_eq_steps
    else:
        data["sul_steps"] = data["steps_eq_oracle"]
    return _create_dfa_from_aalpy_automaton(learned_dfa), data


class _AbstractIntersectionLearner():
    def __init__(self, individual_automata, learner_uses_cache=False, cache_sul_queries=False):
        assert len(individual_automata) > 0, "Automata list may not be empty"
        self._automata = individual_automata
        self._alphabet = individual_automata[0].inputs
        self._learner_uses_cache = learner_uses_cache
        self._dfa_learner = DFALearner(learner_uses_cache, cache_sul_queries)

    def run(self):
        raise NotImplementedError("This abstract learner has not been implemented!")

    def _intersect_automata(self, auts):
        res = auts[0];
        [res := res & x for x in auts[1:]]
        return dfa.utils.minimize(res)

    def _learn_automaton(self, aut, filter=None):
        return self._dfa_learner.run(aut, filter)

    def _learn_automata(self, auts, lazy=True):
        return self._dfa_learner.run(auts, lazy=lazy)

    def reset(self):
        self._dfa_learner.reset()

    @property
    def stats(self):
        return self._dfa_learner.stats


class IndependentLearner(_AbstractIntersectionLearner):
    def run(self):
        # For each automaton, learn the language
        _learned_automata = [self._learn_automaton(aut) for aut in self._automata]
        return self._intersect_automata(_learned_automata)

    def __str__(self):
        return "Indep"


class EagerWordByWordLearner(_AbstractIntersectionLearner):
    def run(self):
        return dfa.utils.minimize(self._learn_automata(self._automata, lazy=False))

    def __str__(self):
        return "EagerWBW"


class LazyWordByWordLearner(_AbstractIntersectionLearner):
    def run(self):
        return dfa.utils.minimize(self._learn_automata(self._automata))

    def __str__(self):
        return "LazyWBW"


class MachineByMachineLearner(_AbstractIntersectionLearner):
    def run(self):
        hyp = make_universal_automaton(self._alphabet);
        [hyp := self._learn_automaton(x, filter=hyp) for x in self._automata]
        return dfa.utils.minimize(hyp)

    def __str__(self):
        return "MBM"
