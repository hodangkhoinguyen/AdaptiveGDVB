import logging
import copy
import numpy as np

from enum import Enum, auto

from fractions import Fraction as F
from pathlib import Path

from gdvb.core.verification_benchmark import VerificationBenchmark

from .evo_step import EvoStep

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
        # TODO: only support one verifier at a time
        self.verifier = list(
            seed_benchmark.settings.verification_configs["verifiers"].values()
        )[0][0]
        self.evo_configs = seed_benchmark.settings.evolutionary
        self.evo_params = self.evo_configs["parameters"]
        self.explore_iter = self.evo_configs["explore_iterations"]
        self.refine_iter = self.evo_configs["refine_iterations"]
        self.explore_cra = (
            self.evo_configs["explore_cra"]
            if "explore_cra" in self.evo_configs
            else False
        )
        self.refine_cra = (
            self.evo_configs["refine_cra"]
            if "refine_cra" in self.evo_configs
            else False
        )
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

            evo_step = self.evolve(evo_step)
            self.benchmarks += [evo_step]

        self.logger.info("EvoGDVB finished successfully!")

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
        )
        return next_evo_step

    def init_explore(self):
        initial_step = EvoStep(
            self.seed_benchmark,
            self.evo_params,
            EvoStep.Direction.Both,
            0,
            self.logger,
            self.explore_cra,
        )
        self.benchmarks += [initial_step]
        return initial_step

    def explore(self, evo_step):
        # compute exploration actions
        direction = evo_step.direction
        if self.state == self.EvoState.Explore:
            if direction == EvoStep.Direction.Both:
                def_rate = self.evo_configs["deflation_rate"]
                inf_rate = self.evo_configs["inflation_rate"]
            elif direction == EvoStep.Direction.Down:
                def_rate = self.evo_configs["deflation_rate"]
                inf_rate = self.evo_configs["deflation_rate"]
            elif direction == EvoStep.Direction.Up:
                def_rate = self.evo_configs["inflation_rate"]
                inf_rate = self.evo_configs["inflation_rate"]
            else:
                assert False

            actions = np.zeros([len(self.evo_params), 2])
            for i, f in enumerate(evo_step.evo_params):
                if self.pivots_ua_found(f) and self.pivots_oa_found(f):
                    actions[i][0] = 1
                    actions[i][1] = 1
                elif self.pivots_ua_found(f):
                    actions[i][0] = inf_rate
                    actions[i][1] = inf_rate
                elif self.pivots_oa_found(f):
                    actions[i][0] = def_rate
                    actions[i][1] = def_rate
                else:
                    actions[i][0] = def_rate
                    actions[i][1] = inf_rate

        ca_configs = evo_step.benchmark.ca_configs
        ca_configs_next = copy.deepcopy(ca_configs)

        parameters_lower_bounds = self.evo_configs["parameters_lower_bounds"]
        parameters_upper_bounds = self.evo_configs["parameters_upper_bounds"]

        for i, f in enumerate(evo_step.factors):
            f = copy.deepcopy(f)
            start = f.start * F(actions[i][0])
            end = f.end * F(actions[i][1])

            # check hard bounds from evo configs
            if f.type in parameters_lower_bounds:
                start = max(start, F(parameters_lower_bounds[f.type]))
            if f.type in parameters_upper_bounds:
                end = min(end, F(parameters_upper_bounds[f.type]))

            # skip factor-level modification if start >= end
            if start > end:
                self.logger.warn(f"START > END!!! NO MODIFICATION TO FACTOR: {f}")
                continue

            f.set_start_end(start, end)

            start, end, levels = f.get()
            ca_configs_next["parameters"]["level"][f.type] = levels
            ca_configs_next["parameters"]["range"][f.type] = [start, end]
        return ca_configs_next

    def update_pivots(self, evo_step):
        solved = self.res_nb_solved[self.verifier]
        total_problems = evo_step.benchmark.ca_configs["parameters"]["level"]["prop"]

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
                self.logger.warn(
                    f"Interesting. We have two Pivot_Us {best_c}. Using the first one: {best_c[0]}."
                )

            for i, f in enumerate(evo_step.evo_params):
                self.pivots_ua[f] = best_c[0][i]

        # UNSET pivot UA
        # this applies to deflation search
        else:
            for f in evo_step.evo_params:
                self.pivots_ua[f] = None

        ### 2) calculate pivot of UA(under-approximation)
        ## Pivots_O is different than Pivots_U, it is one step off the over-approximation
        for i, f in enumerate(evo_step.evo_params):
            candidates = []
            for x in solved:
                good = True

                for y in solved:
                    if y[i] >= x[i] and solved[y] != 0:
                        good = False

                if good:
                    candidates += [x[i]]

            # SET pivot OA
            if candidates:
                self.pivots_oa[f] = np.min(candidates)
            # UNSAT pivot OA
            else:
                self.pivots_oa[f] = None

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

            self.logger.debug("\t FSEL:", f.type, start, end, levels)
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
            self.logger.debug("\t FSEL:", f.type, start, end, levels)

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
            self.times = {v: {} for v in evo_step.times}

        levels = tuple(f.explicit_levels for f in evo_step.factors)

        # TODO : switch to pandas
        # ???? how to separate ndarray _,_ = np.xxx(x)???
        ids = np.array(np.meshgrid(levels[0], levels[1])).T.reshape(
            -1, len(self.evo_params)
        )

        data = list(evo_step.answers.values())[0]
        data = data.reshape(-1, data.shape[-1])

        data2 = list(evo_step.nb_solved.values())[0]
        data2 = data2.reshape(-1, 1)

        data3 = list(evo_step.times.values())[0]
        data3 = data3.reshape(-1, 1)

        # verifier = list(evo_step.answers)[0]
        verifier = self.verifier
        for i, x in enumerate(ids):
            self.res[verifier][tuple(x)] = data[i]
            self.res_nb_solved[verifier][tuple(x)] = data2[i]
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
            #    pie_scatter = PieScatter2D(evo_step.times[verifier])
            #    pie_scatter.heatmap(ticks_f1, ticks_f2, labels_f1, labels_f2)

            #    pie_scatter.save(
            #        f"{pdf_dir}/hm_{self.state}_{evo_step.iteration}_{evo_step.direction}.pdf"
            #    )

        else:
            # plot two factors with properties: |F| >= 3
            # TODO: update plotter to accept more than two factors
            raise NotImplementedError

        if reset_logger_level:
            self.logger.setLevel(level=logging.DEBUG)
