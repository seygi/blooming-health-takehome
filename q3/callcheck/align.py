"""Deterministic alignment of agent messages to script nodes.

For each agent message we find which scripted questions, scenario `say` lines
and outcome `say` lines it contains, and what text is left over (residual).

Matching works on normalized tokens. A node is split into sentences; its
anchor (first sentence, extended while shorter than MIN_ANCHOR_TOKENS) must
appear in the message as an ordered token subsequence inside a compact
window. Later node sentences are optional: agents often drop the trailing
"Take care." of an outcome, so they are consumed when present but never
required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from callcheck.model import COMPLETE, FlowSpec, Thread
from callcheck.spec import scenario_target

# Ordered token containment (LCS / node tokens) needed for a match. 0.8 lets an
# agent drop or change one word in five (e.g. "it sounds" vs "it's") while
# keeping thread_04's paraphrase "do you have active Medi-Cal coverage right
# now?" (0.36 against q2_active's anchor) out.
MATCH_THRESHOLD = 0.8

# Anchors shorter than this borrow the next node sentence. "Totally
# understand." alone (decline question) is too generic to identify a node.
MIN_ANCHOR_TOKENS = 5

# Search window = node length * slack + pad tokens. Keeps the subsequence
# compact so a short node like "Which plan is that?" cannot be assembled from
# words scattered across a long message.
WINDOW_SLACK = 1.5
WINDOW_PAD = 2

# A message sentence counts as script text (removed from residual) when this
# share of its tokens was consumed by matched nodes. "Take care, and goodbye!"
# has only 2 of 4 tokens from "Take care.", so it stays in residual.
SENTENCE_COVERED = 0.6

_DASHES = re.compile("[\u2012\u2013\u2014\u2015\\-]")
_APOSTROPHES = re.compile("[\u2018\u2019\u02bc'`]")
_NON_WORD = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")
_NODE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _APOSTROPHES.sub("", text)  # "that's" -> "thats"
    text = _DASHES.sub(" ", text)  # "medi-cal" -> "medi cal", em dash -> space
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    norm = normalize(text)
    return norm.split(" ") if norm else []


@dataclass(frozen=True)
class AgentTurnAlignment:
    turn: int
    text: str
    questions: list[str]  # item ids whose scripted question appears in this message
    scenario_says: list[tuple[str, str]]  # (item_id, scenario_id) whose scripted `say` appears
    outcome_says: list[str]  # outcome ids whose `say` appears
    residual: str  # message text left after removing matched script spans, trimmed
    scores: dict[str, float]  # node key ("q:<item>", "s:<item>:<scen>", "o:<outcome>") -> best score


@dataclass(frozen=True)
class Alignment:
    turns: list[AgentTurnAlignment]
    asked: list[tuple[int, str]]  # (turn, item_id) in order
    actual_terminal: str | None
    terminal_turn: int | None


@dataclass(frozen=True)
class _Node:
    key: str
    kind: str  # "q", "s", "o"
    ref: tuple[str, ...]  # (item,), (item, scenario), (outcome,)
    anchor: list[str]
    rest: list[list[str]]


def _node(key: str, kind: str, ref: tuple[str, ...], text: str) -> _Node | None:
    sentences = [tokens(s) for s in _NODE_SENTENCE_SPLIT.split(text.strip())]
    sentences = [s for s in sentences if s]
    if not sentences:
        return None
    anchor = list(sentences[0])
    i = 1
    while len(anchor) < MIN_ANCHOR_TOKENS and i < len(sentences):
        anchor += sentences[i]
        i += 1
    return _Node(key=key, kind=kind, ref=ref, anchor=anchor, rest=sentences[i:])


def _nodes(spec: FlowSpec) -> list[_Node]:
    out: list[_Node | None] = []
    for item in spec.items.values():
        q = item.question
        out.append(_node(f"q:{item.id}", "q", (item.id,), q.text))
        for s in q.scenarios:
            if s.say:
                out.append(_node(f"s:{item.id}:{s.id}", "s", (item.id, s.id), s.say))
    for o in spec.outcomes.values():
        if o.say:
            out.append(_node(f"o:{o.id}", "o", (o.id,), o.say))
    return [n for n in out if n is not None]


def _lcs_positions(node: list[str], window: list[str], offset: int) -> list[int]:
    n, m = len(node), len(window)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if node[i] == window[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    positions: list[int] = []
    i = j = 0
    while i < n and j < m:
        if node[i] == window[j]:
            positions.append(offset + j)
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return positions


def _best_match(node: list[str], msg: list[str]) -> tuple[float, list[int]]:
    """Best ordered containment of `node` in a compact window of `msg`."""
    if not node or not msg:
        return 0.0, []
    width = int(len(node) * WINDOW_SLACK) + WINDOW_PAD
    best: list[int] = []
    for start in range(max(1, len(msg) - width + 1)):
        pos = _lcs_positions(node, msg[start : start + width], start)
        if len(pos) > len(best):
            best = pos
            if len(best) == len(node):
                break
    return len(best) / len(node), best


def _split_message(text: str) -> tuple[list[str], list[str], list[int]]:
    """Original sentences, flat message tokens, sentence index per token."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    flat: list[str] = []
    owner: list[int] = []
    for idx, sentence in enumerate(sentences):
        toks = tokens(sentence)
        flat += toks
        owner += [idx] * len(toks)
    return sentences, flat, owner


def _align_turn(
    turn: int, text: str, nodes: list[_Node], asked_so_far: list[str]
) -> AgentTurnAlignment:
    sentences, msg, owner = _split_message(text)
    scores: dict[str, float] = {}
    candidates: list[tuple[_Node, float, list[int]]] = []
    for node in nodes:
        score, pos = _best_match(node.anchor, msg)
        scores[node.key] = score
        if score >= MATCH_THRESHOLD:
            candidates.append((node, score, pos))

    def recency(node: _Node) -> int:
        # Identical say text exists under two items (q3_coverage `true` and
        # decline `reveals_other_coverage`): prefer the item asked last.
        item = node.ref[0] if node.kind in ("q", "s") else None
        if item is None or item not in asked_so_far:
            return -1
        return len(asked_so_far) - 1 - asked_so_far[::-1].index(item)

    # Overlapping candidates: the one explaining more tokens wins. thread_09:
    # the q3_coverage say (23 tokens) beats other_coverage_no_action (15)
    # although both are fully contained.
    candidates.sort(key=lambda c: (len(c[2]) * c[1], recency(c[0])), reverse=True)
    consumed: set[int] = set()
    spans: list[tuple[int, int]] = []
    accepted: list[tuple[_Node, list[int]]] = []
    for node, _score, pos in candidates:
        lo, hi = pos[0], pos[-1]
        if any(lo <= b and a <= hi for a, b in spans):
            continue
        spans.append((lo, hi))
        consumed.update(pos)
        accepted.append((node, pos))
        for sentence in node.rest:
            s_score, s_pos = _best_match(sentence, msg)
            if s_score >= MATCH_THRESHOLD and not consumed.intersection(s_pos):
                consumed.update(s_pos)

    ordered = [n for n, _pos in sorted(accepted, key=lambda a: a[1][0])]  # message order
    questions = [n.ref[0] for n in ordered if n.kind == "q"]
    scenario_says = [(n.ref[0], n.ref[1]) for n in ordered if n.kind == "s"]
    outcome_says = [n.ref[0] for n in ordered if n.kind == "o"]

    per_sentence_total = [0] * len(sentences)
    per_sentence_used = [0] * len(sentences)
    for idx, s_idx in enumerate(owner):
        per_sentence_total[s_idx] += 1
        if idx in consumed:
            per_sentence_used[s_idx] += 1
    kept = [
        s
        for i, s in enumerate(sentences)
        if per_sentence_total[i] == 0
        or per_sentence_used[i] / per_sentence_total[i] < SENTENCE_COVERED
    ]
    return AgentTurnAlignment(
        turn=turn,
        text=text,
        questions=questions,
        scenario_says=scenario_says,
        outcome_says=outcome_says,
        residual=" ".join(kept).strip(),
        scores=scores,
    )


def node_sentences(text: str) -> list[str]:
    """A scripted say split into its sentences, as the matcher sees them."""
    return [s.strip() for s in _NODE_SENTENCE_SPLIT.split(text.strip()) if tokens(s)]


def contains(sentence: str, text: str) -> bool:
    """True when `sentence` appears in `text` under the same matching rule as script nodes."""
    score, _pos = _best_match(tokens(sentence), tokens(text))
    return score >= MATCH_THRESHOLD


def align(spec: FlowSpec, thread: Thread) -> Alignment:
    nodes = _nodes(spec)
    turns: list[AgentTurnAlignment] = []
    asked: list[tuple[int, str]] = []
    actual_terminal: str | None = None
    terminal_turn: int | None = None
    for message in thread.agent_messages():
        ta = _align_turn(message.turn, message.text, nodes, [i for _, i in asked])
        turns.append(ta)
        asked += [(message.turn, item) for item in ta.questions]
        if actual_terminal is not None:
            continue
        if ta.outcome_says:
            actual_terminal, terminal_turn = ta.outcome_says[0], message.turn
            continue
        for item_id, scen_id in ta.scenario_says:
            question = spec.items[item_id].question
            scenario = question.scenario(scen_id)
            if scenario is not None and scenario_target(question, scenario) == COMPLETE:
                actual_terminal, terminal_turn = COMPLETE, message.turn
                break
    return Alignment(
        turns=turns,
        asked=asked,
        actual_terminal=actual_terminal,
        terminal_turn=terminal_turn,
    )


def alignment_of(ctx: Any) -> Alignment:
    """Alignment from a check Context, computing it when the caller did not."""
    if getattr(ctx, "alignment", None) is not None:
        return ctx.alignment
    return align(ctx.spec, ctx.thread)
