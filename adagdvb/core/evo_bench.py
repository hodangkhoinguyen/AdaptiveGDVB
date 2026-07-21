import logging
import copy
import numpy as np

from enum import Enum, auto
from statistics import NormalDist

from fractions import Fraction as F
from pathlib import Path

from gdvb.core.verification_benchmark import VerificationBenchmark

from .evo_step import EvoStep
from .factor import Factor

from gdvb.plot.pie_scatter import PieScatter2D


class EvoBench:
    class EvoState(Enum):
        Explore = auto()
        Refine = auto()

    def __init__(self, seed_benchmark):
        self.logger = seed_benchmark.settings.logger
        self.seed_benchmark = seed_benchmark
        self.benchmark_name = seed_benchmark.settings.name
        self.dnn_configs = seed_benchmark.settings.dnn_configs

        all_verifiers = [
            v
            for tool_verifiers in seed_benchmark.settings.verification_configs[
                "verifiers"
            ].values()
            for v in tool_verifiers
        ]

        self.evo_configs = seed_benchmark.settings.evolutionary
        self.evo_params = self.evo_configs["parameters"]
        self.explore_iter = self.evo_configs["explore_iterations"]
        self.refine_iter = self.evo_configs["refine_iterations"]
        self.explore_cra = self.evo_configs.get("explore_cra", False)
        self.refine_cra = self.evo_configs.get("refine_cra", False)

        # ---- differential VPB (contribution #4) ----
        # Off by default: with no `differential_verifiers` configured, behavior
        # is identical to before (single verifier, taken from the config).
        diff_verifiers = self.evo_configs.get("differential_verifiers")
        if diff_verifiers:
            assert len(diff_verifiers) == 2, (
                "evolutionary.differential_verifiers must list exactly two "
                "verifier names"
            )
            for v in diff_verifiers:
                assert v in all_verifiers, (
                    f"differential_verifiers entry {v!r} is not among the "
                    f"configured verify.verifiers {all_verifiers}"
                )
            self.mode = "differential"
            self.verifiers = list(diff_verifiers)
        else:
            self.mode = "single"
            self.verifiers = [all_verifiers[0]]
        # Backward compatible: single-mode code paths and plotting use this as
        # "the" verifier; in differential mode it is the primary/plotted one.
        self.verifier = self.verifiers[0]

        answer_code = seed_benchmark.settings.answer_code

        # (optional) which conclusive verifier answers count as "solved" for
        # VPB search purposes. Default (unset): sat or unsat, matching the
        # original behavior. A sat instance (falsified by finding a
        # counterexample) tends to resolve quickly regardless of network
        # size, so it dilutes the solve-rate signal used to locate the
        # boundary; setting `success_answers = ['unsat']` focuses the search
        # on proving difficulty specifically, which is where the interesting
        # scaling behavior for a complete verifier actually is.
        self.success_answers = tuple(
            self.evo_configs.get("success_answers", ["sat", "unsat"])
        )
        for a in self.success_answers:
            assert a in answer_code, f"Unknown evolutionary.success_answers entry: {a!r}"
        self._solved_codes = {answer_code[a] for a in self.success_answers}
        self._unsat_code = answer_code["unsat"]
        self._sat_code = answer_code["sat"]

        # (optional) audit every pair of verifiers that actually produced
        # results each iteration for genuine sat/unsat contradictions (one
        # verifier proves the property, the other finds a counterexample to
        # it, on the exact same network+property) -- as opposed to a mere
        # capability gap (one solves, the other times out). Independent of
        # `differential_verifiers`/`mode`: it runs over every verifier listed
        # under [verify.verifiers], not just the two driving the search, and
        # never changes the search itself, only logs what it finds. A
        # contradiction is not proof of a soundness bug on its own -- it can
        # also come from floating-point precision at the property boundary
        # or a benchmark-generation issue -- so it is reported, not treated
        # as a verified defect.
        self.check_consistency = self.evo_configs.get("check_consistency", False)

        # ---- search strategy (contribution #2) ----
        # Default 'geometric' reproduces the original inflation/deflation
        # search exactly. 'active' opts into crossing-estimation search.
        self.search_strategy = self.evo_configs.get("search_strategy", "geometric")
        assert self.search_strategy in ("geometric", "active"), (
            f"Unknown evolutionary.search_strategy: {self.search_strategy}"
        )
        self.active_shrink_rate = self.evo_configs.get("active_shrink_rate", 0.5)

        # ---- probabilistic boundary (contribution #3) ----
        # boundary_threshold unset -> legacy exact-match / magic-number pivot
        # logic (unchanged default). Setting it, choosing search_strategy =
        # 'active', or using differential mode all opt into the continuous,
        # threshold-based pivot search instead.
        self.boundary_threshold = self.evo_configs.get("boundary_threshold")
        self.effective_threshold = (
            self.boundary_threshold if self.boundary_threshold is not None else 0.5
        )

        # (optional) require statistical confidence before letting a point's
        # solve-rate estimate count as evidence in the pivot search: with a
        # small nb_property sample, a point-estimate rate is noisy and can
        # flip a pivot between iterations just from sampling noise near the
        # threshold. Unset (default): compare point estimates directly, as
        # above. Set to a confidence level in (0, 1): use a Wilson score
        # interval on each point's solve-rate estimate, and only treat a
        # point as confirmed-passing/confirmed-failing when the interval
        # clears the threshold with that confidence; ambiguous points
        # contribute no evidence either way.
        self.boundary_confidence = self.evo_configs.get("boundary_confidence")
        assert self.boundary_confidence is None or 0 < self.boundary_confidence < 1, (
            "evolutionary.boundary_confidence must be in (0, 1)"
        )

        # ---- adaptive property-sample budget (contribution #5) ----
        # Off by default: with `nb_property` (the `prop` CA factor level)
        # fixed, every grid point gets the same number of verification calls
        # whether it is deep in "always solves", deep in "never solves", or
        # sitting right on the boundary -- wasting budget in the first two
        # cases and potentially under-sampling the third. When enabled, the
        # `prop` level for the *next* iteration is grown when this
        # iteration's grid has ambiguous points (Wilson interval straddles
        # `boundary_threshold`) and eased back down otherwise, so extra
        # verifier calls are spent only where the boundary actually needs
        # more evidence to resolve.
        self.adaptive_prop_budget = self.evo_configs.get("adaptive_prop_budget", False)
        if self.adaptive_prop_budget:
            assert self.boundary_confidence is not None, (
                "evolutionary.adaptive_prop_budget requires "
                "evolutionary.boundary_confidence, since ambiguity is "
                "defined via the Wilson interval"
            )
            # The legacy pivot rule (exact-full-solve UA / hardcoded ">2
            # solved" OA) and the accumulated per-point rate/bound maps both
            # assume `prop` is constant across the whole Explore phase; that
            # assumption is exactly what this feature breaks, so require one
            # of the modes that already tracks a per-point sample total.
            assert (
                self.mode == "differential"
                or self.boundary_threshold is not None
                or self.search_strategy == "active"
            ), (
                "evolutionary.adaptive_prop_budget requires "
                "boundary_threshold, search_strategy = 'active', or "
                "differential_verifiers -- the legacy pivot rule assumes a "
                "constant prop count across iterations"
            )
            base_prop = seed_benchmark.ca_configs["parameters"]["level"]["prop"]
            self.prop_budget_min = self.evo_configs.get("prop_budget_min", base_prop)
            self.prop_budget_max = self.evo_configs.get("prop_budget_max", base_prop * 4)
            self.prop_budget_growth = self.evo_configs.get("prop_budget_growth", 1.5)
            assert 1 <= self.prop_budget_min <= self.prop_budget_max
            assert self.prop_budget_growth > 1

        # ---- adaptive step size (contribution #6a) ----
        # Default 'static' reproduces the original behavior: inflation_rate/
        # deflation_rate are applied verbatim every iteration, regardless of
        # how far the current data actually is from the boundary. 'adaptive'
        # instead treats them as *ceilings*: the step actually taken shrinks
        # toward 1x as the current iteration's extreme observed solve rate
        # approaches effective_threshold, so a run that is already near the
        # transition zone takes a small, careful step instead of blindly
        # jumping past it, while a run still deep in "always solves"/"always
        # fails" territory still gets the full configured jump.
        self.step_strategy = self.evo_configs.get("step_strategy", "static")
        assert self.step_strategy in ("static", "adaptive"), (
            f"Unknown evolutionary.step_strategy: {self.step_strategy}"
        )
        self.step_sharpness = self.evo_configs.get("step_sharpness", 1.0)
        assert self.step_sharpness > 0, "evolutionary.step_sharpness must be > 0"

        # ---- corner-probe warm start (contribution #6b) ----
        # Off by default: iteration 0 uses the config's `ca.parameters.range`
        # verbatim, as before. When enabled, a cheap probe verifies 3 points
        # along the diagonal of the box defined by parameters_lower_bounds/
        # parameters_upper_bounds -- the two corners and their midpoint (at
        # a small `warm_start_probe_prop` sample count, not the full config
        # budget) -- *before* iteration 0, and iteration 0's actual window
        # is seeded from that real signal instead of a blind config guess.
        # Relies only on the same range+level Factor machinery every other
        # iteration already uses -- no new GDVB/SwarmHost code path.
        self.warm_start = self.evo_configs.get("warm_start", False)
        if self.warm_start:
            base_prop = seed_benchmark.ca_configs["parameters"]["level"]["prop"]
            self.warm_start_probe_prop = self.evo_configs.get(
                "warm_start_probe_prop", min(base_prop, 2)
            )
            assert self.warm_start_probe_prop >= 1
            # half-width of the seeded iteration-0 window, as a fraction of
            # the full [lower_bound, upper_bound] box per factor
            self.warm_start_half_width = self.evo_configs.get(
                "warm_start_half_width", 0.15
            )
            assert 0 < self.warm_start_half_width <= 0.5

        assert len(self.evo_params) == 2
        self._init_parameters()

    def _init_parameters(self):
        self.state = self.EvoState.Explore
        self.benchmarks = []
        self.pivots_ua = {}  # under-approximation
        self.pivots_oa = {}  # over-approximation
        self.res = {}

        for p in self.evo_params:
            self.pivots_ua[p] = None
            self.pivots_oa[p] = None

    def run(self):
        # init explore
        self.logger.info(f"----------[Exploration]----------")
        evo_step = self.init_explore()
        while evo_step:
            self.logger.info(f"----------[Iteration {evo_step.iteration}]----------")
            self.logger.debug(
                f"\t{evo_step.direction}, factors: {[str(x) for x in evo_step.factors]}"
            )

            if not evo_step.load_cache():
                evo_step.forward()
                evo_step.evaluate()
                evo_step.save_cache()
                evo_step.plot()

            self.collect_res(evo_step)
            self.plot(evo_step)
            if self.check_consistency:
                self._check_verifier_consistency(evo_step)

            evo_step = self.evolve(evo_step)
            self.benchmarks += [evo_step]

        self.logger.info("EvoGDVB finished successfully!")

    def _check_verifier_consistency(self, evo_step):
        verifiers = list(evo_step.answers.keys())
        if len(verifiers) < 2:
            return

        found_any = False
        for i in range(len(verifiers)):
            for j in range(i + 1, len(verifiers)):
                v_a, v_b = verifiers[i], verifiers[j]
                a = evo_step.answers[v_a]
                b = evo_step.answers[v_b]
                contradiction = ((a == self._unsat_code) & (b == self._sat_code)) | (
                    (a == self._sat_code) & (b == self._unsat_code)
                )
                count = int(contradiction.sum())
                if count == 0:
                    continue
                found_any = True
                locations = np.argwhere(contradiction).tolist()
                self.logger.warning(
                    f"\tCONSISTENCY CHECK: {v_a} and {v_b} gave opposite conclusive "
                    f"answers (one sat, one unsat) on the same network+property at "
                    f"{count} location(s) in iteration {evo_step.iteration} "
                    f"(factor-level indices, property index): "
                    f"{locations[:10]}{' ...' if count > 10 else ''}. This does not "
                    f"by itself prove a soundness bug -- it can also come from "
                    f"floating-point precision at the property boundary or a "
                    f"benchmark-generation issue -- but it is worth investigating "
                    f"manually."
                )
        if not found_any:
            self.logger.debug(
                f"\tConsistency check: no sat/unsat contradictions among "
                f"{len(verifiers)} verifier(s) in iteration {evo_step.iteration}."
            )

    def evolve(self, evo_step):
        # A1: Exploration State
        if self.state == self.EvoState.Explore:
            # update pivots in the Exploration state
            self.update_pivots(evo_step)
            # generate ca configs
            next_ca_configs = self.explore(evo_step)

            ua_str = f"\tUA:   {self.pivots_ua_found()}"
            oa_str = f"\tOA:   {self.pivots_oa_found()}"
            for x in self.evo_params:
                ua_str += f", {self.pivots_ua[x]}"
                oa_str += f", {self.pivots_oa[x]}"
            self.logger.debug(ua_str + oa_str)

            explore_limit = self.check_same_ca_configs(
                evo_step.benchmark.ca_configs, next_ca_configs
            )

            # goto Refine if both pivots are found
            #             or runs out of predefined exploration steps
            if (
                (self.pivots_oa_found() and self.pivots_ua_found())
                or evo_step.iteration >= self.explore_iter
                or explore_limit
            ):
                self.state = self.EvoState.Refine
                self.logger.info("Exploration finished successfully!")
                self.logger.info(f"----------[Refinement]----------")
                evo_step = self.init_refine(self.benchmarks[0])

        # A2 : Refinement State
        if self.state == self.EvoState.Refine:
            if evo_step.iteration >= self.refine_iter:
                self.logger.info("Refinement finished successfully!")
                return None
            else:
                next_ca_configs = self.refine(evo_step)

        next_benchmark = VerificationBenchmark(
            f"{self.benchmark_name}_{evo_step.iteration+1}_{evo_step.direction}",
            self.dnn_configs,
            next_ca_configs,
            evo_step.benchmark.settings,
        )
        if self.state == self.EvoState.Explore:
            cra = self.explore_cra
        elif self.state == self.EvoState.Refine:
            cra = self.refine_cra
        else:
            assert False
        next_evo_step = EvoStep(
            next_benchmark,
            self.evo_params,
            evo_step.direction,
            evo_step.iteration + 1,
            self.logger,
            cra,
            self.success_answers,
        )
        return next_evo_step

    def init_explore(self):
        initial_benchmark = self.seed_benchmark
        if self.warm_start:
            initial_ca_configs = self._warm_start_ca_configs()
            initial_benchmark = VerificationBenchmark(
                f"{self.benchmark_name}_0_{EvoStep.Direction.Both}",
                self.dnn_configs,
                initial_ca_configs,
                self.seed_benchmark.settings,
            )

        initial_step = EvoStep(
            initial_benchmark,
            self.evo_params,
            EvoStep.Direction.Both,
            0,
            self.logger,
            self.explore_cra,
            self.success_answers,
        )
        self.benchmarks += [initial_step]
        return initial_step

    def _warm_start_ca_configs(self):
        # Cheap corner-and-midpoint probe: verify only 3 points along the
        # (lo,lo) -> (mid,mid) -> (hi,hi) diagonal of the box defined by
        # parameters_lower_bounds/parameters_upper_bounds (a level=3 grid
        # per factor gives this diagonal for free, alongside the 6 off-
        # diagonal points, which are computed but not read here), at a small
        # `warm_start_probe_prop` sample count, and use that real signal to
        # seed iteration 0's window instead of the config's blind
        # `ca.parameters.range` guess.
        #
        # The midpoint matters: with only the two corners, the crossing
        # estimate is a 2-point linear interpolation that collapses to the
        # box midpoint whenever *both* corners happen to be fully saturated
        # (rate exactly 0 or 1) -- which is the common case, since far-out
        # corners tend to have every sampled property agree. A 3rd point
        # lets a saturated-corners read still localize to whichever half of
        # the diagonal actually brackets the crossing, instead of silently
        # degrading to a blind midpoint guess.
        #
        # Uses only the single primary verifier (self.verifier) for the read
        # -- this is a coarse warm-start heuristic, not a pivot decision, so
        # it does not need differential mode's two-verifier care.
        lower = self.evo_configs["parameters_lower_bounds"]
        upper = self.evo_configs["parameters_upper_bounds"]
        threshold = self.effective_threshold
        fallback = copy.deepcopy(self.seed_benchmark.ca_configs)

        probe_ca_configs = copy.deepcopy(self.seed_benchmark.ca_configs)
        for p in self.evo_params:
            probe_ca_configs["parameters"]["level"][p] = 3
            probe_ca_configs["parameters"]["range"][p] = [lower[p], upper[p]]
        probe_ca_configs["parameters"]["level"]["prop"] = self.warm_start_probe_prop

        probe_benchmark = VerificationBenchmark(
            f"{self.benchmark_name}_warmstart_probe",
            self.dnn_configs,
            probe_ca_configs,
            self.seed_benchmark.settings,
        )
        probe_step = EvoStep(
            probe_benchmark,
            self.evo_params,
            EvoStep.Direction.Both,
            "warmstart",
            self.logger,
            False,
            self.success_answers,
        )
        if not probe_step.load_cache():
            probe_step.forward()
            probe_step.evaluate()
            probe_step.save_cache()

        # index 0/1/2 is the lo/mid/hi level, for each of the two evo_params
        # axes (Factor: level=3 over [lo, hi] yields the 3 explicit levels
        # [lo, (lo+hi)/2, hi], in that order); the diagonal is (0,0), (1,1),
        # (2,2).
        rates = (
            np.asarray(probe_step.nb_solved[self.verifier], dtype=np.float64)
            / self.warm_start_probe_prop
        )
        rate_lo_lo, rate_mid, rate_hi_hi = rates[0, 0], rates[1, 1], rates[2, 2]

        if rate_lo_lo < threshold:
            self.logger.warning(
                f"\tWarm-start probe: even the smallest allowed architecture "
                f"has solve rate {rate_lo_lo:.2f} < threshold {threshold:.2f} -- "
                f"the UA pivot may not exist within parameters_lower_bounds/"
                f"parameters_upper_bounds. Falling back to the configured "
                f"initial range."
            )
            return fallback
        if rate_hi_hi >= threshold:
            self.logger.warning(
                f"\tWarm-start probe: even the largest allowed architecture "
                f"has solve rate {rate_hi_hi:.2f} >= threshold {threshold:.2f} -- "
                f"the OA pivot may not exist within parameters_lower_bounds/"
                f"parameters_upper_bounds. Falling back to the configured "
                f"initial range."
            )
            return fallback

        # 3-point piecewise-linear crossing estimate along the diagonal,
        # parameterized by t in [0, 1] from (lo,lo) to (hi,hi).
        diag = [(0.0, rate_lo_lo), (0.5, rate_mid), (1.0, rate_hi_hi)]
        t = 0.5
        for (t0, y0), (t1, y1) in zip(diag, diag[1:]):
            if y0 == threshold:
                t = t0
                break
            if (y0 - threshold) * (y1 - threshold) < 0:
                t = t0 + (threshold - y0) / (y1 - y0) * (t1 - t0)
                break
        else:
            # already checked rate_lo_lo >= threshold > rate_hi_hi above, so
            # a bracketing segment always exists; this is an unreachable
            # fallback, kept only for defensiveness against float edge cases.
            t = 0.5
        t = min(max(t, 0.0), 1.0)
        t_frac = F(t).limit_denominator(1000)

        self.logger.info(
            f"\tWarm-start probe: diagonal rates lo/lo={rate_lo_lo:.2f} "
            f"mid={rate_mid:.2f} hi/hi={rate_hi_hi:.2f} -> seeding "
            f"iteration 0 at diagonal fraction t={t:.2f}"
        )

        fc_conv_ids = {
            "fc": self.seed_benchmark.fc_ids,
            "conv": self.seed_benchmark.conv_ids,
        }
        half_width = F(self.warm_start_half_width).limit_denominator(1000)
        ca_configs_next = copy.deepcopy(self.seed_benchmark.ca_configs)
        for p in self.evo_params:
            lo, hi = F(lower[p]), F(upper[p])
            level = self.seed_benchmark.ca_configs["parameters"]["level"][p]
            center = lo + t_frac * (hi - lo)
            width = (hi - lo) * half_width
            start = max(lo, center - width)
            end = min(hi, center + width)

            factor = Factor(p, lo, hi, level, fc_conv_ids)
            factor.set_start_end(start, end)
            start, end, levels = factor.get()

            ca_configs_next["parameters"]["level"][p] = levels
            ca_configs_next["parameters"]["range"][p] = [start, end]

        return ca_configs_next

    # ---------------- pivot search ----------------

    def update_pivots(self, evo_step):
        total_problems = evo_step.benchmark.ca_configs["parameters"]["level"]["prop"]

        if self.mode == "differential":
            self.pivots_ua, self.pivots_oa = self._differential_pivots(evo_step)
            disagreement = self._disagreement_rate_map(total_problems)
            if disagreement:
                rates = list(disagreement.values())
                self.logger.info(
                    f"\tVerifier disagreement rate ({self.verifiers[0]} vs. "
                    f"{self.verifiers[1]}): max={max(rates):.3f}, "
                    f"mean={float(np.mean(rates)):.3f}"
                )
        elif self.boundary_threshold is not None or self.search_strategy == "active":
            self.pivots_ua, self.pivots_oa = self._verifier_pivots(
                self.verifier, evo_step.evo_params
            )
        else:
            self._update_pivots_legacy(evo_step, total_problems)

    def _update_pivots_legacy(self, evo_step, total_problems):
        # Original pivot search: UA requires an exact full solve, OA uses a
        # hardcoded "> 2 solved" cutoff. Preserved verbatim as the default so
        # existing configs/results are unaffected.
        solved = self.res_nb_solved[self.verifier]

        ### 1) calculate pivot of UA(under-approximation)
        candidates = []
        for i, f in enumerate(evo_step.evo_params):
            can = []
            for x in solved:
                good = True
                for y in solved:
                    # doesn't care problems that are larger
                    if all([y[j] > x[j] for j in range(len(evo_step.evo_params))]):
                        pass
                    elif solved[y] != total_problems:
                        good = False
                if good:
                    can += [x]
            candidates += can

        c = np.array(candidates).reshape(-1, len(evo_step.evo_params)).tolist()
        c = list(set([tuple(x) for x in c]))

        # SET pivot UA
        if c:
            cp = [np.prod(x) for x in c]

            best_c = [x for x in c if np.prod(x) == np.max(cp)]

            if len(best_c) > 1:
                self.logger.warning(
                    f"Interesting. We have two Pivot_Us {best_c}. Using the first one: {best_c[0]}."
                )

            for i, f in enumerate(evo_step.evo_params):
                if self.pivots_ua[f] == None or best_c[0][i] > self.pivots_ua[f]:
                    self.pivots_ua[f] = best_c[0][i]

        # UNSET pivot UA
        # this applies to deflation search
        else:
            for f in evo_step.evo_params:
                self.pivots_ua[f] = None

        ### 2) calculate pivot of OA(over-approximation)
        ## Pivots_O is different than Pivots_U, it is one step off the over-approximation
        for i, f in enumerate(evo_step.evo_params):
            candidates = []
            for x in solved:
                good = True

                for y in solved:
                    if y[i] >= x[i] and solved[y] > 2:
                        good = False

                if good:
                    candidates += [x[i]]

            # SET pivot OA
            if candidates:
                self.pivots_oa[f] = np.min(candidates)
            # UNSAT pivot OA
            else:
                self.pivots_oa[f] = None

    def _rate_map(self, verifier):
        # Divides each point by its own recorded sample size rather than a
        # single shared `total_problems`, since `adaptive_prop_budget` (see
        # __init__) can make the `prop` count differ between the iteration
        # that first recorded a point and the current one.
        solved = self.res_nb_solved[verifier]
        totals = self.res_total_problems[verifier]
        return {
            point: float(np.asarray(count).reshape(-1)[0]) / totals[point]
            for point, count in solved.items()
        }

    def _confidence_bounds_map(self, verifier):
        # Per-point (lower, upper) Wilson score bound on the true solve rate,
        # given a sample of the point's own recorded total with `count`
        # solved (see _rate_map on why this is per-point, not shared).
        solved = self.res_nb_solved[verifier]
        totals = self.res_total_problems[verifier]
        confidence = self.boundary_confidence
        return {
            point: self._wilson_interval(
                float(np.asarray(count).reshape(-1)[0]), totals[point], confidence
            )
            for point, count in solved.items()
        }

    @staticmethod
    def _wilson_interval(k, n, confidence):
        if n <= 0:
            return 0.0, 1.0
        z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
        p_hat = k / n
        denom = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        margin = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    def _disagreement_rate_map(self, total_problems):
        # Region of interest for differential VPB: per-property fraction of
        # properties where the two verifiers' solved/unsolved status differs.
        v_a, v_b = self.verifiers
        answers_a = self.res[v_a]
        answers_b = self.res[v_b]

        rate_map = {}
        for point in answers_a:
            if point not in answers_b:
                continue
            a = np.asarray(answers_a[point])
            b = np.asarray(answers_b[point])
            solved_a = np.isin(a, list(self._solved_codes))
            solved_b = np.isin(b, list(self._solved_codes))
            rate_map[point] = float(np.mean(solved_a != solved_b))
        return rate_map

    def _differential_pivots(self, evo_step):
        # Disagreement rate is not monotone in the factors (it is ~0 where
        # both verifiers agree-solve, rises in the transition zone, and
        # falls back to ~0 where both agree-fail), so it can't be searched
        # directly with the monotone box-dominance pivot search below. Each
        # verifier's own solve rate IS monotone, though, so instead we pivot
        # each verifier separately and take the union of their transition
        # zones: below min(UA_A, UA_B) both verifiers still fully solve
        # (agreement), and beyond max(OA_A, OA_B) both have failed
        # (agreement again) -- so this union is guaranteed to contain every
        # point where the two verifiers can possibly disagree.
        threshold = self.effective_threshold
        pivot_ua = {p: None for p in evo_step.evo_params}
        pivot_oa = {p: None for p in evo_step.evo_params}
        for verifier in self.verifiers:
            ua, oa = self._verifier_pivots(verifier, evo_step.evo_params)
            for p in evo_step.evo_params:
                if ua[p] is not None:
                    pivot_ua[p] = ua[p] if pivot_ua[p] is None else min(pivot_ua[p], ua[p])
                if oa[p] is not None:
                    pivot_oa[p] = oa[p] if pivot_oa[p] is None else max(pivot_oa[p], oa[p])
        return pivot_ua, pivot_oa

    def _verifier_pivots(self, verifier, evo_params):
        # One verifier's own threshold-based pivots, using confidence bounds
        # instead of point estimates when boundary_confidence is set.
        if self.boundary_confidence is not None:
            bounds_map = self._confidence_bounds_map(verifier)
            return self._confidence_threshold_pivots(
                bounds_map, evo_params, self.effective_threshold
            )
        rate_map = self._rate_map(verifier)
        return self._threshold_pivots(rate_map, evo_params, self.effective_threshold)

    @staticmethod
    def _threshold_pivots(rate_map, evo_params, threshold):
        # Generalizes the legacy pivot search: replaces the exact "fully
        # solved" match (UA) and the hardcoded ">2 solved" cutoff (OA) with a
        # single, configurable solve-rate threshold. Assumes rate_map is
        # monotone non-increasing in every factor (true for a single
        # verifier's solve rate; NOT true for a raw disagreement rate, see
        # _differential_pivots).
        points = list(rate_map.keys())
        passes = lambda x: rate_map[x] >= threshold
        fails = lambda x: rate_map[x] < threshold
        return EvoBench._dominance_pivots(points, passes, fails, evo_params)

    @staticmethod
    def _confidence_threshold_pivots(bounds_map, evo_params, threshold):
        # Same dominance search, but a point only counts as evidence
        # ("passes"/"fails") once its Wilson interval clears the threshold
        # with the configured confidence; points whose interval straddles
        # the threshold are ambiguous and contribute no evidence either way,
        # which keeps a few noisy trials from prematurely setting or
        # flip-flopping a pivot.
        points = list(bounds_map.keys())
        passes = lambda x: bounds_map[x][0] >= threshold
        fails = lambda x: bounds_map[x][1] < threshold
        return EvoBench._dominance_pivots(points, passes, fails, evo_params)

    @staticmethod
    def _dominance_pivots(points, passes, fails, evo_params):
        n = len(evo_params)

        pivot_ua = {p: None for p in evo_params}
        ua_candidates = []
        for x in points:
            if not passes(x):
                continue
            good = True
            for y in points:
                if all(y[j] > x[j] for j in range(n)):
                    continue
                if fails(y):
                    good = False
                    break
            if good:
                ua_candidates += [x]
        if ua_candidates:
            best = max(ua_candidates, key=lambda x: np.prod(x))
            for i, p in enumerate(evo_params):
                pivot_ua[p] = best[i]

        pivot_oa = {p: None for p in evo_params}
        for i, p in enumerate(evo_params):
            candidates = [
                x[i] for x in points if not any(y[i] >= x[i] and passes(y) for y in points)
            ]
            if candidates:
                pivot_oa[p] = min(candidates)
        return pivot_ua, pivot_oa

    # ---------------- range search strategies ----------------

    def explore(self, evo_step):
        if self.search_strategy == "active":
            next_ca_configs = self._explore_active(evo_step)
        else:
            next_ca_configs = self._explore_geometric(evo_step)
        if self.adaptive_prop_budget:
            next_ca_configs["parameters"]["level"]["prop"] = self._next_prop_budget(evo_step)
        return next_ca_configs

    def _next_prop_budget(self, evo_step):
        # Uses this iteration's own arrays directly (not the accumulated,
        # multi-iteration self.res_nb_solved) so the ambiguity read is always
        # against a single well-defined sample size: the `prop` level this
        # iteration actually ran with.
        total_problems = evo_step.benchmark.ca_configs["parameters"]["level"]["prop"]
        threshold = self.effective_threshold
        confidence = self.boundary_confidence

        ambiguous_fractions = []
        for verifier in self.verifiers:
            counts = np.asarray(evo_step.nb_solved[verifier], dtype=np.float64).reshape(-1)
            if counts.size == 0:
                continue
            bounds = [self._wilson_interval(c, total_problems, confidence) for c in counts]
            ambiguous = sum(1 for lo, hi in bounds if lo < threshold <= hi)
            ambiguous_fractions += [ambiguous / len(bounds)]
        ambiguous_fraction = max(ambiguous_fractions) if ambiguous_fractions else 0.0

        if ambiguous_fraction > 0:
            next_prop = min(
                self.prop_budget_max, int(round(total_problems * self.prop_budget_growth))
            )
        else:
            next_prop = max(
                self.prop_budget_min, int(round(total_problems / self.prop_budget_growth))
            )

        if next_prop != total_problems:
            self.logger.info(
                f"\tAdaptive prop budget: {ambiguous_fraction:.0%} of grid points "
                f"ambiguous under {confidence:.0%} confidence at prop={total_problems} "
                f"-> next prop={next_prop}"
            )
        return next_prop

    def _direction_rates(self, evo_step):
        direction = evo_step.direction
        if direction == EvoStep.Direction.Both:
            def_cap, inf_cap = self.evo_configs["deflation_rate"], self.evo_configs["inflation_rate"]
        elif direction == EvoStep.Direction.Down:
            def_cap, inf_cap = self.evo_configs["deflation_rate"], self.evo_configs["deflation_rate"]
        elif direction == EvoStep.Direction.Up:
            def_cap, inf_cap = self.evo_configs["inflation_rate"], self.evo_configs["inflation_rate"]
        else:
            raise ValueError(f"Unknown explore direction: {direction}")

        if self.step_strategy != "adaptive":
            return def_cap, inf_cap
        return self._adaptive_direction_rates(evo_step, def_cap, inf_cap)

    def _adaptive_direction_rates(self, evo_step, def_cap, inf_cap):
        # Scales each cap by how far this iteration's extreme observed solve
        # rate is from effective_threshold, using this iteration's own arrays
        # only (not the accumulated multi-iteration res_nb_solved), so the
        # signal always corresponds to a single well-defined prop count.
        threshold = self.effective_threshold
        total_problems = evo_step.benchmark.ca_configs["parameters"]["level"]["prop"]

        rates = []
        for verifier in self.verifiers:
            counts = np.asarray(evo_step.nb_solved[verifier], dtype=np.float64).reshape(-1)
            if counts.size:
                rates += (counts / total_problems).tolist()
        if not rates:
            return def_cap, inf_cap

        # Deflation direction: no fully-solving point found yet, so the
        # relevant signal is the *best* rate seen. Near 1 -> still deep in
        # "solves everywhere", take the full configured step; near threshold
        # -> the transition may already be inside the window, take a small,
        # careful step instead of overshooting past it.
        #
        # Normalized against the *achievable* distance, not a raw [0, 1]
        # span: since rate is bounded in [0, 1], the farthest max(rates) can
        # ever get from threshold is (1 - threshold), and the farthest
        # min(rates) can get is threshold itself. Without this normalization
        # the full configured cap would be unreachable whenever threshold
        # isn't 0 or 1 (e.g. rate=1.0 against threshold=0.5 is only "0.5
        # away" on a raw scale, not the "maximally far" it actually is).
        def_span = max(1e-9, 1 - threshold)
        d_def = min(1.0, abs(max(rates) - threshold) / def_span)
        def_rate = 1 + (def_cap - 1) * (d_def ** self.step_sharpness)

        # Inflation direction: mirrors the above using the *worst* rate seen.
        inf_span = max(1e-9, threshold)
        d_inf = min(1.0, abs(min(rates) - threshold) / inf_span)
        inf_rate = 1 + (inf_cap - 1) * (d_inf ** self.step_sharpness)

        return def_rate, inf_rate

    def _factor_action(self, factor_type, def_rate, inf_rate):
        if self.pivots_ua_found(factor_type) and self.pivots_oa_found(factor_type):
            return 1, 1
        elif self.pivots_ua_found(factor_type):
            return inf_rate, inf_rate
        elif self.pivots_oa_found(factor_type):
            return def_rate, def_rate
        else:
            return def_rate, inf_rate

    def _explore_geometric(self, evo_step):
        def_rate, inf_rate = self._direction_rates(evo_step)

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)

        parameters_lower_bounds = self.evo_configs["parameters_lower_bounds"]
        parameters_upper_bounds = self.evo_configs["parameters_upper_bounds"]

        for f in evo_step.factors:
            f = copy.deepcopy(f)
            a0, a1 = self._factor_action(f.type, def_rate, inf_rate)
            start = f.start * F(a0)
            end = f.end * F(a1)

            # check hard bounds from evo configs
            if f.type in parameters_lower_bounds:
                start = max(start, F(parameters_lower_bounds[f.type]))
            if f.type in parameters_upper_bounds:
                end = min(end, F(parameters_upper_bounds[f.type]))

            # skip factor-level modification if start >= end
            if start > end:
                self.logger.warning(f"START > END!!! NO MODIFICATION TO FACTOR: {f}")
                continue

            f.set_start_end(start, end)

            start, end, levels = f.get()
            ca_configs_next["parameters"]["level"][f.type] = levels
            ca_configs_next["parameters"]["range"][f.type] = [start, end]
        return ca_configs_next

    def _explore_active(self, evo_step):
        # Active-search alternative to blind geometric doubling/halving: fits
        # a joint monotone solve-rate surface over both factors from all data
        # collected so far (contribution #6), estimates where each factor's
        # conditional slice through that surface crosses `effective_
        # threshold`, and centers a shrinking window there. Falls back to
        # the geometric action for a factor when there isn't yet a
        # bracketing crossing to estimate from.
        #
        # Fitting jointly (see _monotonize_rate_map) instead of per-axis
        # marginal averaging matters whenever the two factors interact --
        # e.g. a diagonal boundary like "solves iff neu*fc < K" -- since
        # averaging a factor's crossing over whatever values of the *other*
        # factor happened to be sampled blends together crossings that
        # differ per slice, misplacing the estimate for either one. Reading
        # each factor's crossing off the slice nearest the current window's
        # center for the other factor (_slice_crossing) avoids that blend.
        #
        # In differential mode there is no single monotone surface to cross
        # (see _differential_pivots), so each verifier's own crossing is
        # estimated separately and the window is the span between them,
        # which brackets the disagreement region directly without needing an
        # extra shrink-width guess.
        threshold = self.effective_threshold

        if self.mode == "differential":
            rate_maps = [self._rate_map(v) for v in self.verifiers]
        else:
            rate_maps = [self._rate_map(self.verifier)]
        fitted_maps = [self._monotonize_rate_map(rm) for rm in rate_maps]

        def_rate, inf_rate = self._direction_rates(evo_step)

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)
        parameters_lower_bounds = self.evo_configs["parameters_lower_bounds"]
        parameters_upper_bounds = self.evo_configs["parameters_upper_bounds"]
        shrink = F(self.active_shrink_rate).limit_denominator(1000)

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)
            # exactly 2 evo_params is asserted in __init__
            other_idx = 1 - i
            other_factor = evo_step.factors[other_idx]
            other_ref = float((other_factor.start + other_factor.end) / 2)
            crossings = [
                c
                for c in (
                    self._slice_crossing(fm, i, other_idx, other_ref, threshold)
                    for fm in fitted_maps
                )
                if c is not None
            ]

            if not crossings:
                a0, a1 = self._factor_action(f.type, def_rate, inf_rate)
                start = f.start * F(a0)
                end = f.end * F(a1)
            elif len(crossings) > 1:
                start, end = min(crossings), max(crossings)
                if start == end:
                    min_step = f.min_step if f.min_step is not None else F(0)
                    half = max((f.end - f.start) * shrink, min_step) / 2
                    start, end = start - half, end + half
            else:
                min_step = f.min_step if f.min_step is not None else F(0)
                width = max((f.end - f.start) * shrink, min_step)
                half = width / 2
                start = crossings[0] - half
                end = crossings[0] + half

            if f.type in parameters_lower_bounds:
                start = max(start, F(parameters_lower_bounds[f.type]))
            if f.type in parameters_upper_bounds:
                end = min(end, F(parameters_upper_bounds[f.type]))

            if start > end:
                self.logger.warning(f"START > END!!! NO MODIFICATION TO FACTOR: {f}")
                continue

            f.set_start_end(start, end)
            start, end, levels = f.get()
            ca_configs_next["parameters"]["level"][f.type] = levels
            ca_configs_next["parameters"]["range"][f.type] = [start, end]
        return ca_configs_next

    @staticmethod
    def _monotonize_rate_map(rate_map):
        # Projects a possibly noisy, scattered 2D solve-rate sample onto
        # values consistent with the assumed monotonicity (rate
        # non-increasing in every factor): repeatedly clips each point's
        # rate into [max rate among points that dominate it, min rate among
        # points it dominates] until nothing changes. Dominating points
        # (equal-or-harder in every factor) lower-bound a point's rate;
        # points it dominates (equal-or-easier in every factor) upper-bound
        # it. This is a simple bounding-envelope projection -- not a
        # globally L2-optimal isotonic regression (that would need PAVA over
        # a full rectangular grid or a QP solver) -- but it is cheap
        # (O(n^2) per pass, no new dependency), works directly on the
        # scattered, cross-iteration-accumulated point set _rate_map
        # actually produces, and is the piece that replaces per-axis
        # marginal averaging with a genuinely joint fit.
        points = list(rate_map.keys())
        n = len(points)
        fitted = dict(rate_map)
        for _ in range(max(1, n)):
            changed = False
            next_fitted = dict(fitted)
            for x in points:
                dominating = [
                    fitted[y]
                    for y in points
                    if y != x and all(y[j] >= x[j] for j in range(len(x)))
                ]
                dominated = [
                    fitted[y]
                    for y in points
                    if y != x and all(y[j] <= x[j] for j in range(len(x)))
                ]
                value = fitted[x]
                if dominating:
                    value = max(value, max(dominating))
                if dominated:
                    value = min(value, min(dominated))
                if value != fitted[x]:
                    changed = True
                next_fitted[x] = value
            fitted = next_fitted
            if not changed:
                break
        return fitted

    @staticmethod
    def _slice_crossing(fitted_map, factor_idx, other_idx, other_ref, threshold):
        # Reads factor_idx's crossing off the slice of the joint fit nearest
        # other_ref (the current window's center for the other factor),
        # instead of averaging over every sampled value of the other factor
        # -- the fix for a diagonal/interacting boundary (see
        # _explore_active).
        other_values = {x[other_idx] for x in fitted_map}
        if not other_values:
            return None
        other_val = min(other_values, key=lambda v: abs(v - other_ref))

        slice_points = {
            x[factor_idx]: rate
            for x, rate in fitted_map.items()
            if x[other_idx] == other_val
        }
        if len(slice_points) < 2:
            return None

        xs = sorted(slice_points)
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = slice_points[x0], slice_points[x1]
            if y0 == threshold:
                return F(float(x0)).limit_denominator(10000)
            if (y0 - threshold) * (y1 - threshold) < 0:
                t = (threshold - y0) / (y1 - y0)
                crossing = x0 + t * (x1 - x0)
                return F(float(crossing)).limit_denominator(10000)
        return None

    def init_refine(self, evo_step):
        self.logger.info("\tInitialize refinement stage.")

        # clean previous benchmark results for refinement phase
        self.res = None
        self.res_nb_solved = None

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)

        for f in evo_step.factors:
            f = copy.deepcopy(f)

            start = self.pivots_ua[f.type]
            end = self.pivots_oa[f.type]

            if not start:
                start = F(self.evo_configs["parameters_lower_bounds"][f.type])
            if not end:
                end = F(self.evo_configs["parameters_upper_bounds"][f.type])

            f.set_start_end(start, end)
            start, end, levels = f.get()

            self.logger.debug(f"\t FSEL: {f.type} {start} {end} {levels}")
            ca_configs_next["parameters"]["level"][f.type] = levels
            ca_configs_next["parameters"]["range"][f.type] = [start, end]

        next_benchmark = VerificationBenchmark(
            f"{self.benchmark_name}_{evo_step.iteration}_{EvoStep.Direction.Maintain}",
            self.dnn_configs,
            ca_configs_next,
            evo_step.benchmark.settings,
        )
        next_evo_step = EvoStep(
            next_benchmark,
            self.evo_params,
            EvoStep.Direction.Maintain,
            0,
            self.logger,
            self.refine_cra,
            self.success_answers,
        )

        return next_evo_step

    def refine(self, evo_step):
        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)
        arity = self.evo_configs["refine_arity"]

        for f in evo_step.factors:
            f = copy.deepcopy(f)
            f.subdivision(arity)

            start, end, levels = f.get()
            self.logger.debug(f"\t FSEL: {f.type} {start} {end} {levels}")

            ca_configs_next["parameters"]["level"][f.type] = levels
            ca_configs_next["parameters"]["range"][f.type] = [start, end]
        return ca_configs_next

    def pivots_oa_found(self, f=None):
        if f:
            found = self.pivots_oa[f] is not None
        else:
            found = all([self.pivots_oa[x] is not None for x in self.pivots_oa])
        return found

    def pivots_ua_found(self, f=None):
        if f:
            found = self.pivots_ua[f] is not None
        else:
            found = all([self.pivots_ua[x] is not None for x in self.pivots_ua])
        return found

    def check_same_ca_configs(self, this, that):
        res = []
        for p in self.evo_params:
            this_start = F(this["parameters"]["range"][p][0])
            that_start = F(that["parameters"]["range"][p][0])
            this_end = F(this["parameters"]["range"][p][1])
            that_end = F(that["parameters"]["range"][p][1])
            this_level = F(this["parameters"]["level"][p])
            that_level = F(that["parameters"]["level"][p])

            res += [this_start == that_start]
            res += [this_end == that_end]
            res += [this_level == that_level]

        # print(res, all(x for x in res))
        return all(x for x in res)

    def collect_res(self, evo_step):
        if not self.res or self.state == self.EvoState.Refine:
            self.res = {v: {} for v in evo_step.answers}
            self.res_nb_solved = {v: {} for v in evo_step.answers}
            # per-point sample size (the `prop` level the point was actually
            # verified with), recorded alongside the count since
            # adaptive_prop_budget can vary it between iterations
            self.res_total_problems = {v: {} for v in evo_step.answers}
            self.times = {v: {} for v in evo_step.times}

        total_problems = evo_step.benchmark.ca_configs["parameters"]["level"]["prop"]
        levels = tuple(f.explicit_levels for f in evo_step.factors)

        # TODO : switch to pandas
        # ???? how to separate ndarray _,_ = np.xxx(x)???
        ids = np.array(np.meshgrid(levels[0], levels[1])).T.reshape(
            -1, len(self.evo_params)
        )

        # collect data for every verifier this EvoBench tracks (one in single
        # mode, two in differential mode), keyed explicitly by verifier name
        # rather than assuming the first entry of evo_step.answers.
        for verifier in self.verifiers:
            data = evo_step.answers[verifier]
            data = data.reshape(-1, data.shape[-1])

            data2 = evo_step.nb_solved[verifier]
            data2 = data2.reshape(-1, 1)

            data3 = evo_step.times[verifier]
            data3 = data3.reshape(-1, 1)

            for i, x in enumerate(ids):
                self.res[verifier][tuple(x)] = data[i]
                self.res_nb_solved[verifier][tuple(x)] = data2[i]
                self.res_total_problems[verifier][tuple(x)] = total_problems
                self.times[verifier][tuple(x)] = data3[i]

    # plot two factors with properties: |F| = 3
    # TODO: update plotter to accept more than two factors
    def plot(self, evo_step):
        if self.logger.level == logging.DEBUG:  # debug level
            self.logger.setLevel(level=logging.INFO)
            reset_logger_level = True
        else:
            reset_logger_level = False

        if len(self.evo_params) == 2:
            labels = [x for x in self.evo_params]
            ticks = {x: set() for x in self.evo_params}

            # verifier = list(self.benchmarks[0].answers)[0]
            verifier = self.verifier
            ticks = np.array(
                [list(x) for x in self.res[verifier].keys()], dtype=np.float32
            )
            data = np.array([x for x in self.res[verifier].values()], dtype=np.float32)

            data2 = np.array(
                [x for x in self.times[verifier].values()], dtype=np.float32
            )

            # print(self.evo_params[0], set(sorted(np.array([list(x) for x in self.res[verifier].keys()])[:, 0].tolist())))
            # print(self.evo_params[1], set(sorted(np.array([list(x) for x in self.res[verifier].keys()])[:, 1].tolist())))

            ticks_f1 = ticks[:, 0].tolist()
            ticks_f2 = ticks[:, 1].tolist()

            labels_f1 = labels[0]
            labels_f2 = labels[1]

            pdf_dir = f"{self.seed_benchmark.settings.root}/figures_{verifier}/"
            Path(pdf_dir).mkdir(parents=True, exist_ok=True)

            pie_scatter = PieScatter2D(data)

            pie_scatter.draw_with_ticks(
                ticks_f1,
                ticks_f2,
                labels_f1,
                labels_f2,
                legend_size=20,
                label_size=30,
                tick_size=20,
            )
            pie_scatter.save(
                f"{pdf_dir}/all_{self.state}_{evo_step.iteration}_{evo_step.direction}.pdf"
            )

            pie_scatter.draw_with_ticks(
                ticks_f1,
                ticks_f2,
                labels_f1,
                labels_f2,
                x_log_scale=True,
                y_log_scale=True,
                legend_size=20,
                label_size=30,
                tick_size=20,
                display_legend=True,
            )
            pie_scatter.save(
                f"{pdf_dir}/all_log_{self.state}_{evo_step.iteration}_{evo_step.direction}.pdf"
            )

            # if self.verifier == "neurify" and self.explore_iter == 0:
            if self.state == self.EvoState.Refine:
                pie_scatter = PieScatter2D(evo_step.times[verifier])
                pie_scatter.heatmap(ticks_f1, ticks_f2, labels_f1, labels_f2)

                pie_scatter.save(
                    f"{pdf_dir}/hm_{self.state}_{evo_step.iteration}_{evo_step.direction}.pdf"
                )

        else:
            # plot two factors with properties: |F| >= 3
            # TODO: update plotter to accept more than two factors
            raise NotImplementedError

        if reset_logger_level:
            self.logger.setLevel(level=logging.DEBUG)
