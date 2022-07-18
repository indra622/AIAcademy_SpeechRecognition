from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional

import math

import torch

import k2

import re,logging, sys

Lexicon = List[Tuple[str, List[str]]]

def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """It adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.
    See also add_lex_disambig.pl from kaldi.
    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
    Returns:
      Return a tuple with two elements:
        - The output lexicon with disambiguation symbols
        - The ID of the max disambiguation symbol that appears
          in the lexicon
    """

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))
    return ans, max_disambig


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """Read a lexicon from `filename`.
    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).
    Args:
      filename:
        Path to the lexicon.txt
    Returns:
      A list of tuples., e.g., [('w', ['p1', 'p2']), ('w1', ['p3, 'p4'])]
    """
    ans = []

    with open(filename, "r", encoding="utf-8") as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                logging.info(
                    f"Found bad line {line} in lexicon file {filename}"
                )
                logging.info(
                    "Every line is expected to contain at least 2 fields"
                )
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(
                    f"Found bad line {line} in lexicon file {filename}"
                )
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]
            ans.append((word, tokens))

    return ans


def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """Generate ID maps, i.e., map a symbol to a unique ID.
    Args:
      symbols:
        A list of unique symbols.
    Returns:
      A dict containing the mapping between symbols and IDs.
    """
    return {sym: i for i, sym in enumerate(symbols)}


def lexicon_to_fst(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).
    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    # The blank symbol <blk> is defined in local/train_bpe_model.py
    #assert token2id["<blk>"] == 0
    #assert word2id["<eps>"] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [token2id[i] for i in pieces]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )
        #disambig_token = 0
        #disambig_word = 0
        #arcs = add_self_loops(
        #    arcs,
        #    disambig_token=disambig_token,
        #    disambig_word=disambig_word,
        #)

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa

def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.
    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.
    The input label of a self-loop is `disambig_token`, while the output
    label is `disambig_word`.
    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_token:
        It is the token ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.
    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_token, disambig_word, 0])

    return arcs + ans


def save_kaldi_styles(dictionary, filename):
    with open(filename, 'w') as f:
        for k, v in dictionary.items():
            f.write(str(k)+' '+str(v)+'\n')
            
            
            
class Nbest(object):
    """
    An Nbest object contains two fields:
        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
                 Its axes are [path][state][arc]
        (2) shape. Its type is :class:`k2.RaggedShape`.
                   Its axes are [utt][path]
    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.
    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        """
        Args:
          fsa:
            An FsaVec with axes [path][state][arc]. It is expected to contain
            a list of **linear** FSAs.
          shape:
            A ragged shape with two axes [utt][path].
        """
        assert len(fsa.shape) == 3, f"fsa.shape: {fsa.shape}"
        assert shape.num_axes == 2, f"num_axes: {shape.num_axes}"

        if fsa.shape[0] != shape.tot_size(1):
            raise ValueError(
                f"{fsa.shape[0]} vs {shape.tot_size(1)}\n"
                "Number of FSAs in `fsa` does not match the given shape"
            )

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = "Nbest("
        s += f"Number of utterances:{self.shape.dim0}, "
        s += f"Number of Paths:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        nbest_scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.
        Each sampled path is a linear FSA.
        We assume `lattice.labels` contains token IDs and `lattice.aux_labels`
        contains word IDs.
        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
          use_double_scores:
            True to use double precision in :func:`k2.random_paths`.
            False to use single precision.
          scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
        Returns:
          Return an Nbest instance.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= nbest_scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos]
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        # It axes is [utt][path][word_id]
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)
        word_seq = word_seq.remove_values_leq(0)

        # Each utterance has `num_paths` paths but some of them transduces
        # to the same word sequence, so we need to remove repeated word
        # sequences within an utterance. After removing repeats, each utterance
        # contains different number of paths
        #
        # `new2old` is a 1-D torch.Tensor mapping from the output path index
        # to the input path index.
        _, _, new2old = word_seq.unique(
            need_num_repeats=False, need_new2old_indexes=True
        )

        # kept_path is a ragged tensor with dtype torch.int32.
        # It has axes [utt][path][arc_pos]
        kept_path, _ = path.index(new2old, axis=1, need_value_indexes=False)

        # utt_to_path_shape has axes [utt][path]
        utt_to_path_shape = kept_path.shape.get_layer(0)

        # Remove the utterance axis.
        # Now kept_path has only two axes [path][arc_pos]
        kept_path = kept_path.remove_axis(0)

        # labels is a ragged tensor with 2 axes [path][token_id]
        # Note that it contains -1s.
        labels = k2.ragged.index(lattice.labels.contiguous(), kept_path)

        # Remove -1 from labels as we will use it to construct a linear FSA
        labels = labels.remove_values_eq(-1)

        if isinstance(lattice.aux_labels, k2.RaggedTensor):
            # lattice.aux_labels is a ragged tensor with dtype torch.int32.
            # It has 2 axes [arc][word], so aux_labels is also a ragged tensor
            # with 2 axes [arc][word]
            aux_labels, _ = lattice.aux_labels.index(
                indexes=kept_path.values, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.values)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        # Caution: fsa.scores are all 0s.
        # `fsa` has only one extra attribute: aux_labels.
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice, get 1-best
        path from the resulting FsaVec, and return a new Nbest object.
        The purpose of this function is to attach scores to an Nbest.
        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume its `labels` are word IDs.
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`. Also, its `fsa` has non-zero scores and inherits attributes
          for `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels

        word_fsa.scores.zero_()
        word_fsa_with_epsilon_loops = k2.remove_epsilon_and_add_self_loops(
            word_fsa
        )

        path_to_utt_map = self.shape.row_ids(1)

        if hasattr(lattice, "aux_labels"):
            # lattice has token IDs as labels and word IDs as aux_labels.
            # inv_lattice has word IDs as labels and token IDs as aux_labels
            inv_lattice = k2.invert(lattice)
            inv_lattice = k2.arc_sort(inv_lattice)
        else:
            inv_lattice = k2.arc_sort(lattice)

        if inv_lattice.shape[0] == 1:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=torch.zeros_like(path_to_utt_map),
                sorted_match_a=True,
            )
        else:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=path_to_utt_map,
                sorted_match_a=True,
            )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        path_lattice = k2.top_sort(k2.connect(path_lattice))

        one_best = k2.shortest_path(
            path_lattice, use_double_scores=use_double_scores
        )

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).
        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).
        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.
        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]
        am_scores = self.fsa.scores - self.fsa.lm_scores
        ragged_am_scores = k2.RaggedTensor(scores_shape, am_scores.contiguous())
        tot_scores = ragged_am_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def compute_lm_scores(self) -> k2.RaggedTensor:
        """Compute LM scores of each linear FSA (i.e., each path within
        an utterance).
        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).
        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.
        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_lm_scores = k2.RaggedTensor(
            scores_shape, self.fsa.lm_scores.contiguous()
        )

        tot_scores = ragged_lm_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of FSAs in this Nbest.
        Note:
          Since FSAs in Nbest are just linear FSAs, log-semiring
          and tropical semiring produce the same total scores.
        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_scores = k2.RaggedTensor(
            scores_shape, self.fsa.scores.contiguous()
        )

        tot_scores = ragged_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def build_levenshtein_graphs(self) -> k2.Fsa:
        """Return an FsaVec with axes [utt][state][arc]."""
        word_ids = get_texts(self.fsa, return_ragged=True)
        return k2.levenshtein_graph(word_ids)


def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()