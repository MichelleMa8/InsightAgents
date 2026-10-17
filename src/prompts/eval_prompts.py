"""
Pairwise LLM-judge prompt: CoI-Agent's five aspects, each with a comparative
decision rule, plus shared fairness rules and a tagged output format with a
short rationale per aspect.
"""


# The five substantive dimensions carry the research-contribution criteria.
# Each is judged on its own; there is no holistic summary aspect, which would
# tend to dominate the five rather than follow from them.
EVAL_ASPECTS = ["novelty", "significance", "feasibility", "clarity", "effectiveness"]

# Fairness constraints.
_JUDGING_RULES = """General rules for all criteria (very important):
- Judge CONTENT, not presentation. Length, number of sections, bullet density, formatting polish, and confident tone are NOT evidence of quality on any criterion. Prefer the idea that is specific and straight-to-the-point over one that is longer or more elaborately formatted.
- Do not reward claims you cannot verify. Citations, numbers, or assertions about prior work that you cannot check count neither for nor against the idea; they never raise novelty or effectiveness on their own.
- Keep the criteria orthogonal. Judge each criterion only against its own definition: do not lower feasibility because an idea is unoriginal, and do not raise novelty because an idea is well written.
- Every idea has limitations. Only limitations that bear on the criterion you are currently judging should affect that criterion.
- State of the art is not required. Impact may come from efficiency, generalizability, interpretability, robustness, or fairness, not only from beating a leaderboard."""

# Comparative decision rules, one per aspect.
_CRITERIA_PAIR = """1. Novelty: Which idea makes the more substantive original intellectual contribution: an insight, hypothesis, problem formulation, method, or explanatory connection? A meaningful combination of familiar techniques can be original when their interaction enables new understanding or capability. Mere assembly, complexity or renaming is not novelty. Do not favor or penalize an idea because it is easier to implement. Support duplication claims with identifiable prior work when known; acknowledge uncertainty rather than inventing references.
2. Significance: Which central contribution, if established, would teach the field more or address a more consequential bottleneck? Consider changes in understanding, useful capabilities and what others could build on, including focused studies and informative negative results. Judge supported potential value independently of execution difficulty. Breadth and ambitious language alone do not establish significance.
3. Feasibility: Which idea has the more credible path for an academic group to implement its core mechanism and test its central claim? Distinguish essential requirements from optional extensions and manageable adjustments from fundamental barriers. Adjustments must preserve the intellectual contribution; do not invent missing solutions or weaken the claim. Lower cost or shorter duration alone should not decide between two credible plans. Compare essential resources, assumptions and unresolved dependencies; novelty and ambition do not excuse an infeasible plan.
4. Clarity: Which idea more clearly connects its central contribution, mechanism, assumptions and proposed test? Require the details needed to assess that connection, not an exhaustive implementation recipe. Do not favor simpler scope or a longer checklist of datasets and metrics over a precisely explained intellectual contribution. Penalize undefined concepts and missing reasoning, not conceptual ambition or routine implementation choices left open.
5. Effectiveness: Which idea's mechanism and evaluation more credibly deliver its stated research objective and produce justified knowledge? A hypothesis test can be effective even when it refutes the hypothesis; a claimed improvement still needs a plausible mechanism for achieving it. Prefer tests that distinguish the claim from credible alternatives and support the conclusions actually sought. Do not substitute ease of execution or guaranteed positive results for evidential strength. Penalize circular evaluation, confounding and unsupported inferential leaps.
"""


def _output_format(value_desc: str) -> str:
    """Tagged output block: a short rationale before each aspect's verdict."""
    return "\n".join(
        f"<{a}_reason>{{ 1-2 sentences citing specific content of the idea }}</{a}_reason>\n"
        f"<{a}>{{ {value_desc} for {a} }}</{a}>"
        for a in EVAL_ASPECTS
    )


def get_judge_idea_pair_prompt(idea0: str, idea1: str, topic: str) -> str:
    return f"""
You are a judge in a competition. You have to decide which idea is better.

The idea0 is: {idea0}

The idea1 is: {idea1}

The topic is: {topic}

Which idea do you think is better? Judge each criterion separately.

Here are your evaluation criteria:
{_CRITERIA_PAIR}

{_JUDGING_RULES}

Note:
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. DO NOT allow the LENGTH of the responses to influence your evaluation, choose the one that is straight-to-the-point instead of unnecessarily verbose. Be as objective as possible. (very important!!!)

If you think idea0 is better than idea1, you should output 0. If you think idea1 is better than idea0, you should output 1. If you think idea0 and idea1 are equally good, you should output 2. Only output 2 when the two ideas are genuinely equivalent on that criterion - do not use it to avoid making a decision.

Your output should be strictly in following format:
Your thinking process:
...

Your choice:
{_output_format("Your choice (0, 1, or 2)")}
    """

