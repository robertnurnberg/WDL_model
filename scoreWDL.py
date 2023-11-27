import argparse, json, matplotlib.pyplot as plt, numpy as np, time
from ast import literal_eval
from collections import Counter
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, minimize
from typing import Literal, cast


def win_rate(eval: int | np.ndarray, a, b):
    """the win rate in our model is 1 / ( 1 + exp(-(eval - a) / b))"""

    def stable_logistic(z):
        # returns 1 / (1 + exp(-z)) avoiding possible overflows
        if type(z) == np.ndarray:
            mask = z < 0
            res = np.empty_like(z)
            res[mask] = np.exp(z[mask]) / (1.0 + np.exp(z[mask]))
            res[~mask] = 1.0 / (1.0 + np.exp(-z[~mask]))
            return res
        return np.exp(z) / (1.0 + np.exp(z)) if z < 0 else 1.0 / (1.0 + np.exp(-z))

    # guard against unphysical values, treating negative values of b as "zero"
    # during optimizations this will guide b (back) towards positive values
    if type(b) == np.ndarray:
        b = np.maximum(b, 1e-8)
    elif b < 1e-8:
        b = 1e-8

    return stable_logistic((eval - a) / b)


def loss_rate(eval: int | np.ndarray, a, b):
    """our wdl model assumes symmetry in eval"""
    return win_rate(-eval, a, b)


def poly3(x: float | np.ndarray, c_3, c_2, c_1, c_0) -> float:
    """compute the value of a polynomial of 3rd order in a point x"""
    return ((c_3 * x + c_2) * x + c_1) * x + c_0


def model_wdl_rates(
    eval: int | np.ndarray,
    mom: int | np.ndarray,
    mom_target: int,
    coeffs_a: list[float],
    coeffs_b: list[float],
) -> tuple[float, float, float]:
    """our wdl model is based on win/loss rate with a and b polynomials in mom,
    where mom = move or material counter"""
    a = poly3(mom / mom_target, *coeffs_a)
    b = poly3(mom / mom_target, *coeffs_b)
    w = win_rate(eval, a, b)
    l = loss_rate(eval, a, b)
    return w, 1 - w - l, l


@dataclass
class ModelDataDensity:
    """Count data converted to densities"""

    xs: list[int]  # internal evals
    ys: list[int]  # mom counters (mom = move or material)
    zwins: list[float]  # corresponding win probabilities
    zdraws: list[float]  # draw probabilities


class WdlData:
    """stores the raw wdl data in three 2D numpy arrays"""

    def __init__(self, args, eval_max):
        self.yData = args.yData
        self.filenames = args.filename
        self.yDataMin = args.yDataMin
        self.yDataMax = args.yDataMax  
        self.NormalizeData = args.NormalizeData
        if self.NormalizeData is not None:
            self.NormalizeData = json.loads(self.NormalizeData)
            self.NormalizeData["as"] = [float(x) for x in self.NormalizeData["as"]]
        self.normalize_to_pawn_value = (
            args.NormalizeToPawnValue
            if self.NormalizeData is None
            else int(sum(self.NormalizeData["as"]))
        )
        print(
            "Converting evals with "
            + (
                f"NormalizeToPawnValue = {self.normalize_to_pawn_value}."
                if self.NormalizeData is None
                else f"NormalizeData = {self.NormalizeData}."
            )
        )

        # numpy arrays have nonnegative indices, so save dims and offsets for later
        # TODO: dim no need to keep
        self.dim_mom = args.yDataMax - args.yDataMin + 1
        self.offset_mom = args.yDataMin
        self.eval_max = round(eval_max * self.normalize_to_pawn_value / 100)
        self.dim_eval = 2 * self.eval_max + 1
        self.offset_eval = -self.eval_max

        # set up three arrays, each counting positions with a given internal mom (move/material) and eval
        # note that contour plots for 2D matrices show row indices on y-axis, hence mom is row index TODO
        # TODO: investigate if sparse matrix data structure is faster overall
        # TODO: check if fortran order is faster
        self.wins = np.zeros((self.dim_eval, self.dim_mom), dtype=int)
        self.draws = np.zeros((self.dim_eval, self.dim_mom), dtype=int)
        self.losses = np.zeros((self.dim_eval, self.dim_mom), dtype=int)

    def idx(self, eval, mom):
        return eval - self.offset_eval, mom - self.offset_mom

    def add_to_wdl_counters(self, result, eval, mom, value):
        eval_idx, mom_idx = self.idx(eval, mom)
        if result == "W":
            self.wins[eval_idx, mom_idx] += value
        elif result == "D":
            self.draws[eval_idx, mom_idx] += value
        elif result == "L":
            self.losses[eval_idx, mom_idx] += value

    def load_json_data(self, move_min, move_max):
        """load the WDL data from json: the keys describe the position (result, move, material, eval),
        and the values are the observed count of these positions"""
        for filename in self.filenames:
            print(f"Reading eval stats from {filename}.")
            with open(filename) as infile:
                data = json.load(infile)

                for key, value in data.items() if data else []:
                    result, move, material, eval = literal_eval(key)

                    if move < move_min or move > move_max:
                        continue

                    mom = move if self.yData == "move" else material

                    # convert the cp eval to the internal value by undoing the normalization
                    if self.NormalizeData is None:
                        # undo static rescaling, that was constant in mom
                        a_internal = self.normalize_to_pawn_value
                    else:
                        # undo dynamic rescaling, that was dependent on mom
                        mom_clamped = min(
                            max(mom, self.NormalizeData["yDataMin"]),
                            self.NormalizeData["yDataMax"],
                        )
                        a_internal = poly3(
                            mom_clamped / self.NormalizeData["yDataTarget"],
                            *self.NormalizeData["as"],
                        )
                    eval_internal = round(eval * a_internal / 100)

                    if abs(eval_internal) <= self.eval_max:
                        self.add_to_wdl_counters(result, eval_internal, mom, value)

        W, D, L = self.wins.sum(), self.draws.sum(), self.losses.sum()
        print(f"Retained (W,D,L) = ({W}, {D}, {L}) positions.")

        if W + D + L == 0:
            print("No data was found!")
            exit(0)

        # define wdl densities: if total = 0, entries will be NaN
        total = self.wins + self.draws + self.losses
        self.mask = total > 0
        self.w_density = np.full_like(total, np.NaN, dtype=float)
        self.d_density = np.full_like(total, np.NaN, dtype=float)
        self.l_density = np.full_like(total, np.NaN, dtype=float)
        self.w_density[self.mask] = self.wins[self.mask] / total[self.mask]
        self.d_density[self.mask] = self.draws[self.mask] / total[self.mask]
        self.l_density[self.mask] = self.losses[self.mask] / total[self.mask]

    def get_wdl_columns(self, mom):
        """return views of the three 2d density arrays for the given value of mom"""
        mom_idx = mom - self.offset_mom  # recover the array index of mom
        eval_mask = self.mask[:, mom_idx]  # find all the evals with wdl data
        row_idxs = np.where(eval_mask)[0] + self.offset_eval  # recover eval values
        w = self.wins[:, mom_idx][eval_mask]
        d = self.draws[:, mom_idx][eval_mask]
        l = self.losses[:, mom_idx][eval_mask]
        return row_idxs, w, d, l

    def get_wdl_density_columns(self, mom):
        """return views of the three 2d density arrays for the given value of mom"""
        mom_idx = mom - self.offset_mom  # recover the array index of mom
        eval_mask = self.mask[:, mom_idx]  # find all the evals with wdl data
        row_idxs = np.where(eval_mask)[0] + self.offset_eval  # recover eval values
        w = self.w_density[:, mom_idx][eval_mask]
        d = self.d_density[:, mom_idx][eval_mask]
        l = self.l_density[:, mom_idx][eval_mask]
        return row_idxs, w, d, l

    def get_model_data_density(self) -> ModelDataDensity:
        """for backward compatibility, to allow easy contour plotting"""
        # TODO: try to remove
        valid_data = np.where(self.mask)
        xs, ys = valid_data[0] + self.offset_eval, valid_data[1] + self.offset_mom
        zwins = self.w_density[valid_data]
        zdraws = self.d_density[valid_data]
        return ModelDataDensity(xs, ys, zwins, zdraws)

    def fit_abs_locally(self, modelFitting):
        """for each value of mom of interest, find a(mom) and b(mom) so that the induced
        1D win rate function best matches the observed win frequencies"""

        # filter mom values with less than 10 wins in total
        total_wins = np.sum(self.wins, axis=0)
        self.mom_mask = total_wins >= 10  # TODO: make cli parameter
        if not np.all(self.mom_mask):
            false_indices = np.where(~self.mom_mask)[0]
            print(
                f"Warning: Too little data, so skipping {self.yData} values",
                np.where(~self.mom_mask)[0] + self.offset_mom,
            )

        # prepare the values of mom for which we will fit a and b locally
        model_ms = np.where(self.mom_mask)[0] + self.offset_mom
        model_as = np.empty_like(model_ms, dtype=float)
        model_bs = np.empty_like(model_ms, dtype=float)

        for i in range(len(model_ms)):
            xdata, ywindata, ydrawdata, ylossdata = self.get_wdl_density_columns(
                model_ms[i]
            )

            # find a(mom) and b(mom) via a simple fit of win_rate() to the densities
            popt_ab = self.normalize_to_pawn_value * np.array([1, 1 / 6])
            popt_ab, _ = curve_fit(win_rate, xdata, ywindata, popt_ab)

            # refine the local result based on data, optimizing an objective function
            if modelFitting != "fitDensity":
                # minimize the objective function
                objective_function = ObjectiveFunction(modelFitting, self, [model_ms[i]])
                popt_ab, _ = objective_function.minimize(popt_ab)

            model_as[i] = popt_ab[0]  # store a(mom)
            model_bs[i] = popt_ab[1]  # store b(mom)

        return model_ms, model_as, model_bs


class ObjectiveFunction:
    """Provides objective functions that can be minimized to fit the win draw loss data"""

    def __init__(
        self,
        modelFitting: str,
        wdl_data: WdlData,
        moms: list[int] | None,
        y_data_target: int = None,
    ):
        if modelFitting == "optimizeScore":
            # minimize the l2 error of the predicted score
            self._objective_function = self.scoreError
        elif modelFitting == "optimizeProbability":
            # maximize the likelihood of predicting the game outcome
            self._objective_function = self.evalLogProbability
        else:
            self._objective_function = None
        self.wdl_data = wdl_data
        self.moms = range(wdl_data.yDataMin, wdl_data.yDataMax + 1) if moms is None else moms
        self.y_data_target = y_data_target

    def get_ab(self, asbs: list[float], mom: int):
        # returns p_a(mom), p_b(mom) or a(mom), b(mom) depending on optimization stage
        if len(asbs) == 8:
            coeffs_a = asbs[0:4]
            coeffs_b = asbs[4:8]
            a = poly3(mom / self.y_data_target, *coeffs_a)
            b = poly3(mom / self.y_data_target, *coeffs_b)
        else:
            a = asbs[0]
            b = asbs[1]

        return a, b

    def estimateScore(self, asbs: list[float], eval: int, mom: int):
        """Estimate game score based on probability of WDL"""

        a, b = self.get_ab(asbs, mom)
        probw = win_rate(eval, a, b)
        probl = loss_rate(eval, a, b)
        probd = 1 - probw - probl
        return probw + 0.5 * probd + 0

    def scoreError(self, asbs: list[float]):
        """Sum of the squared error on the game score"""
        scoreErr = 0
        totalCount = 0

        for mom in self.moms:
            evals, w, d, l = wdl_data.get_wdl_columns(mom)
            for a, score in [(w, 1), (d, 0.5), (l, 0)]:
                for i in range(len(evals)):
                    count = a[i]
                    scoreErr += count * (self.estimateScore(asbs, evals[i], mom) - score) ** 2
                    totalCount += count

        return np.sqrt(scoreErr / totalCount)

    def evalLogProbability(self, asbs: list[float]):
        """-log(product of game outcome probability)"""
        evalLogProb = 0
        totalCount = 0

        for mom in self.moms:
            a, b = self.get_ab(asbs, mom)
            evals, w, d, l = wdl_data.get_wdl_columns(mom)
            for i in range(len(evals)):
                probw = win_rate(evals[i], a, b)
                probl = loss_rate(evals[i], a, b)
                probd = 1 - probw - probl
                evalLogProb += w[i] * np.log(max(probw, 1e-14))
                evalLogProb += d[i] * np.log(max(probd, 1e-14))
                evalLogProb += l[i] * np.log(max(probl, 1e-14))
                totalCount += w[i] + d[i] + l[i]

        return -evalLogProb / totalCount

    def __call__(self, asbs):
        return 0 if self._objective_function is None else self._objective_function(asbs)

    def minimize(self, initial_ab):
        if self._objective_function is None:
            return initial_ab, "No objective function defined, return initial guess."

        res = minimize(
            self._objective_function,
            initial_ab,
            method="Powell",
            options={"maxiter": 100000, "disp": False, "xtol": 1e-6},
        )
        return res.x, res.message

class WdlModel:
    def __init__(self, args):
        self.yDataTarget = args.yDataTarget
        self.modelFitting = args.modelFitting

    def poly3_str(self, coeffs: list[float]) -> str:
        return (
            "((%5.3f * x / %d + %5.3f) * x / %d + %5.3f) * x / %d + %5.3f"
            % tuple(
                val for pair in zip(coeffs, [self.yDataTarget] * 4) for val in pair
            )[:-1]
        )

    def fit_ab_globally(self, wdl_data: WdlData):
        print(f"Fit WDL model based on {wdl_data.yData}.")

        # for each value of mom of interest, find good fits for a(mom) and b(mom)
        self.model_ms, self.model_as, self.model_bs = wdl_data.fit_abs_locally(
            self.modelFitting
        )

        # now capture the functional behavior of a and b as functions of mom,
        # starting with a simple polynomial fit to find p_a and p_b
        self.coeffs_a, _ = curve_fit(
            poly3, self.model_ms / self.yDataTarget, self.model_as
        )
        self.coeffs_b, _ = curve_fit(
            poly3, self.model_ms / self.yDataTarget, self.model_bs
        )

        # possibly refine p_a and p_b by optimizing a given objective function
        if self.modelFitting != "fitDensity":
            objective_function = ObjectiveFunction(
                self.modelFitting, wdl_data, None, self.yDataTarget
            )

            popt_all = self.coeffs_a.tolist() + self.coeffs_b.tolist()
            print("Initial objective function: ", objective_function(popt_all))
            popt_all, message = objective_function.minimize(popt_all)
            self.coeffs_a = popt_all[0:4]  # store final p_a
            self.coeffs_b = popt_all[4:8]  # store final p_b
            print("Final objective function:   ", objective_function(popt_all))
            print(message)

        # prepare output
        self.label_as = "as = " + self.poly3_str(self.coeffs_a)
        self.label_bs = "bs = " + self.poly3_str(self.coeffs_b)

        # now we can report the new conversion factor p_a from internal eval to centipawn
        # such that an expected win score of 50% is for an internal eval of p_a(mom)
        # for a static conversion (independent of mom), we provide a constant value
        # NormalizeToPawnValue = int(p_a(yDataTarget)) = int(sum(coeffs_a))
        fsum_a, fsum_b = sum(self.coeffs_a), sum(self.coeffs_b)

        print(f"const int NormalizeToPawnValue = {int(fsum_a)};")
        print(f"Corresponding spread = {int(fsum_b)};")
        print(f"Corresponding normalized spread = {fsum_b / fsum_a};")
        print(
            f"Draw rate at 0.0 eval at move {self.yDataTarget} = {1 - 2 / (1 + np.exp(fsum_a / fsum_b))};"
        )

        print("Parameters in internal value units: ")
        print(self.label_as + "\n" + self.label_bs)
        print(
            "     constexpr double as[] = {%13.8f, %13.8f, %13.8f, %13.8f};"
            % tuple(self.coeffs_a)
        )
        print(
            "     constexpr double bs[] = {%13.8f, %13.8f, %13.8f, %13.8f };"
            % tuple(self.coeffs_b)
        )


class WdlPlot:
    def __init__(self, args, normalize_to_pawn_value: int):
        self.setting = args.plot
        if self.setting == "no":
            return

        self.pgnName = args.pgnName
        self.normalize_to_pawn_value = normalize_to_pawn_value
        self.yPlotMin = args.yPlotMin

        self.fig, self.axs = plt.subplots(  # set figure size to A4 x 1.5
            2, 3, figsize=(11.69 * 1.5, 8.27 * 1.5), constrained_layout=True
        )
        self.fig.suptitle(
            "Summary of win-draw-loss "
            + ("data" if args.modelFitting == "None" else "model analysis"),
            fontsize="x-large",
        )

    def normalized_axis(self, i, j):
        """provides a second x-axis in pawns, to go with the original axis in internal eval
        if the engine used a dynamic normalization, the labels will only be exact for
        the old yDataTarget value for mom (move or material counter)"""
        eval_min, eval_max = self.axs[i, j].get_xlim()
        halfpawn_value = self.normalize_to_pawn_value / 2
        halfpawn_ticks = np.arange(
            eval_min / halfpawn_value, eval_max / halfpawn_value + 1, dtype=int
        )
        ax2 = self.axs[i, j].twiny()
        ax2.set_xticks(halfpawn_ticks * halfpawn_value)  # ticks at full and half pawns
        ax2.set_xticklabels(["" if z % 2 else str(z // 2) for z in halfpawn_ticks])
        ax2.set_xlim(eval_min, eval_max)  # align the data range with original axis

    def sample_wdl_densities(self, wdl_data, mom):
        """plot wdl sample data at a fixed mom value"""
        xdata, ywindata, ydrawdata, ylossdata = wdl_data.get_wdl_density_columns(mom)
        self.axs[0, 0].plot(xdata, ywindata, "b.", label="Measured winrate")
        self.axs[0, 0].plot(xdata, ydrawdata, "g.", label="Measured drawrate")
        self.axs[0, 0].plot(xdata, ylossdata, "c.", label="Measured lossrate")

        self.axs[0, 0].set_xlabel(
            "Evaluation [lower: Internal Value units, upper: Pawns]"
        )
        self.axs[0, 0].set_ylabel("outcome")
        self.axs[0, 0].legend(fontsize="small")
        self.axs[0, 0].set_title(f"Measured data at {wdl_data.yData} {mom}")
        # plot between -3 and 3 pawns
        xmax = ((3 * self.normalize_to_pawn_value) // 100 + 1) * 100
        self.axs[0, 0].set_xlim([-xmax, xmax])

        self.normalized_axis(0, 0)

    def sample_wdl_curves(self, a, b):
        """add the three wdl model curves to subplot axs[0, 0]"""
        xdata = np.linspace(*self.axs[0, 0].get_xlim(), num=1000)
        winmodel = win_rate(xdata, a, b)
        lossmodel = loss_rate(xdata, a, b)
        self.axs[0, 0].plot(xdata, winmodel, "r-", label="Model")
        self.axs[0, 0].plot(xdata, lossmodel, "r-")
        self.axs[0, 0].plot(xdata, 1 - winmodel - lossmodel, "r-")
        self.axs[0, 0].set_title(
            "Comparison of model and m" + self.axs[0, 0].title.get_text()[1:]
        )

    def contour_plots(self, wdl_data: WdlData, model: WdlModel):
        print("Preparing contour plots.")

        plot_fitted_model = model is not None
        if plot_fitted_model:
            # graphs of a and b as a function of move/material
            self.axs[1, 0].plot(model.model_ms, model.model_as, "b.", label="as")
            self.axs[1, 0].plot(
                model.model_ms,
                poly3(model.model_ms / model.yDataTarget, *model.coeffs_a),
                "r-",
                label="fit: " + model.label_as,
            )
            self.axs[1, 0].plot(model.model_ms, model.model_bs, "g.", label="bs")
            self.axs[1, 0].plot(
                model.model_ms,
                poly3(model.model_ms / model.yDataTarget, *model.coeffs_b),
                "m-",
                label="fit: " + model.label_bs,
            )

            self.axs[1, 0].set_xlabel(wdl_data.yData)
            self.axs[1, 0].set_ylabel("parameters (in internal value units)")
            self.axs[1, 0].legend(fontsize="x-small")
            self.axs[1, 0].set_title("Winrate model parameters")
            self.axs[1, 0].set_ylim(bottom=0.0)

        model_data_density = wdl_data.get_model_data_density()

        # now generate contour plots
        contourlines = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0]

        ylabelStr = wdl_data.yData + " (1,3,3,5,9)" * bool(wdl_data.yData == "material")
        ymin, ymax = self.yPlotMin, wdl_data.yDataMax
        points = np.array(list(zip(model_data_density.xs, model_data_density.ys)))

        for j, j_str in enumerate(["win", "draw"]):
            # for wins, plot between -1 and 3 pawns, for draws between -2 and 2 pawns
            xmin = -(((1 + j) * wdl_data.normalize_to_pawn_value) // 100 + 1) * 100
            xmax = (((3 - j) * wdl_data.normalize_to_pawn_value) // 100 + 1) * 100
            grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]  # use a 30x22 grid

            for i, i_str in enumerate(
                ["Data", "Model"] if plot_fitted_model else ["Data"]
            ):
                self.axs[i, 1 + j].yaxis.grid(True)
                self.axs[i, 1 + j].xaxis.grid(True)
                self.axs[i, 1 + j].set_xlabel(
                    "Evaluation [lower: Internal Value units, upper: Pawns]"
                )
                self.axs[i, 1 + j].set_ylabel(ylabelStr)

                zz: np.ndarray | list[float]
                if i_str == "Data":
                    zz = model_data_density.zdraws if j else model_data_density.zwins
                else:
                    zz = cast(
                        np.ndarray,
                        model_wdl_rates(
                            np.asarray(model_data_density.xs),
                            np.asarray(model_data_density.ys),
                            model.yDataTarget,
                            model.coeffs_a,
                            model.coeffs_b,
                        )[j],
                    )
                zz = griddata(points, zz, (grid_x, grid_y))
                cp = self.axs[i, 1 + j].contourf(grid_x, grid_y, zz, contourlines)

                CS = self.axs[i, 1 + j].contour(
                    grid_x, grid_y, zz, contourlines, colors="black"
                )
                self.axs[i, 1 + j].clabel(CS, inline=1, colors="black")
                self.axs[i, 1 + j].set_title(
                    i_str + ": Fraction of positions leading to a " + j_str
                )
                self.normalized_axis(i, 1 + j)

        self.fig.colorbar(cp, ax=self.axs[:, -1], shrink=0.6)
        self.fig.align_labels()
        self.save()

    def save(self):
        plt.savefig(self.pgnName, dpi=300)
        if self.setting == "save+show":
            plt.show()
        plt.close()
        print(f"Saved graphics to {self.pgnName}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Stockfish's WDL model to fishtest game statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="*",
        help="json file(s) with fishtest games' WDL statistics",
        default=["scoreWDLstat.json"],
    )
    parser.add_argument(
        "--NormalizeToPawnValue",
        type=int,
        help="Value needed for converting the games' cp evals to the SF's internal eval.",
    )
    parser.add_argument(
        "--NormalizeData",
        type=str,
        help='Allow dynamic conversion. E.g. {"yDataMin": 11, "yDataMax": 120, "yDataTarget": 32, "as": [0.38036525, -2.82015070, 23.17882135, 307.36768407]}.',
    )
    parser.add_argument(
        "--moveMin",
        type=int,
        default=0,
        help="Lower move number limit for filter applied to json data.",
    )
    parser.add_argument(
        "--moveMax",
        type=int,
        default=120,
        help="Upper move number limit for filter applied to json data.",
    )
    parser.add_argument(
        "--yData",
        choices=["move", "material"],
        default="move",
        help="Select y-axis data used for plotting and fitting.",
    )
    parser.add_argument(
        "--yDataMin",
        type=int,
        default=3,
        help="Minimum value of yData to consider for plotting and fitting.",
    )
    parser.add_argument(
        "--yDataMax",
        type=int,
        default=120,
        help="Maximum value of yData to consider for plotting and fitting.",
    )
    parser.add_argument(
        "--yDataTarget",
        type=int,
        default=32,
        help="Value of yData at which new rescaled 100cp should correspond to 50:50 winning chances.",
    )
    parser.add_argument(
        "--yPlotMin",
        type=int,
        help="Overrides --yDataMin for plotting.",
    )
    parser.add_argument(
        "--plot",
        choices=["save+show", "save", "no"],
        default="save+show",
        help="Save/show graphics or not. Useful for batch processing.",
    )
    parser.add_argument(
        "--pgnName",
        default="scoreWDL.png",
        help="Name of saved graphics file.",
    )
    parser.add_argument(
        "--modelFitting",
        choices=["fitDensity", "optimizeProbability", "optimizeScore", "None"],
        default="fitDensity",
        help="Choice of model fitting: Fit the win rate curves, maximimize the probability of predicting the outcome, minimize the squared error in predicted score, or no fitting.",
    )
    parser.add_argument(
        "--usenumpy",
        action="store_true",
    )

    args = parser.parse_args()

    if args.NormalizeToPawnValue is None:
        if args.NormalizeData is None:
            args.NormalizeToPawnValue = 328
    else:
        assert (
            args.NormalizeData is None
        ), "Error: Can only specify one of --NormalizeToPawnValue and --NormalizeData."

    if args.yData == "material":
        # fix default values for material
        if args.yDataMax == 120 and args.yDataMin == 3:
            args.yDataMin, args.yDataMax = 10, 78

    if args.yPlotMin is None:
        args.yPlotMin = (
            max(10, args.yDataMin) if args.yData == "move" else args.yDataMin
        )

    tic = time.time()

    if not args.usenumpy:
        data_loader = DataLoader(args)

        win, draw, loss = data_loader.extract_wdl(
            args.moveMin, args.moveMax, args.yData
        )

        if len(win) + len(draw) + len(loss) == 0:
            print("No data was found!")
            exit(0)

        wdl_plot = WdlPlot(args, data_loader.normalize_to_pawn_value)

        wdl_model = WdlModel(args, data_loader.normalize_to_pawn_value, wdl_plot)

        model_data_density = data_loader.get_model_data_density(win, draw, loss)

        model = wdl_model.fit_model(model_data_density, win, draw, loss)

        if args.plot != "no":
            wdl_plot.create_plots(model_data_density, model)
    else:
        wdl_data = WdlData(args, eval_max=400)  # TODO: make cli parameter
        wdl_data.load_json_data(args.moveMin, args.moveMax)

        if args.modelFitting != "None":
            wdl_model = WdlModel(args)
            wdl_model.fit_ab_globally(wdl_data)
        else:
            wdl_model = None

        if args.plot != "no":
            wdl_plot = WdlPlot(args, wdl_data.normalize_to_pawn_value)
            wdl_plot.sample_wdl_densities(wdl_data, args.yDataTarget)
            if wdl_model:
                # this shows the fit of the observed wdl data at mom=yDataTarget to
                # the model wdl rates with a=p_a(yDataTarget) and b=p_b(yDataTarget)
                fsum_a, fsum_b = sum(wdl_model.coeffs_a), sum(wdl_model.coeffs_b)
                wdl_plot.sample_wdl_curves(fsum_a, fsum_b)

            wdl_plot.contour_plots(wdl_data, wdl_model)

    if args.plot != "save+show":
        print(f"Total elapsed time = {time.time() - tic:.2f}s.")
