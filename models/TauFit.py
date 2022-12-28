from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QWidget

from .Parameter import Parameter, TAU_PARAMETER_NAME
from .Point import Point

from protocols import SettingsSource, Collection

from typing import get_args, Literal
from functools import partial

from scipy.optimize import least_squares # type: ignore
from scipy.linalg import svd #type: ignore

from numpy import power, exp, log, finfo, sqrt, diag, linspace, meshgrid
from numpy import max as np_max

from math import nextafter

from pandas import Series, DataFrame, concat #type: ignore

TauParameters = tuple[Parameter, ...]

class TauFit(QObject):
    """_summary_

        Args:
            name (str): Name of TauFit.
            compound (SettingsSource): Examined compound.
            collection (_type_): The collection to which it belongs.

        Attributes:
            name_changed: Emitted when TauFit name change. Contains new name.
            all_parameters_changed: Emitted when multiple parameters change. 
            points_changed: Emitted when multiple points change.
            varying_changed: Emitted when varying ax in slice change. Contains name of new varying ax.
            constant_changed: Emitted when constant ax in slice chang. Contains value of new constant ax.
            parameters_saved: Emitted when saved parameters change.
    """

    name_changed: pyqtSignal = pyqtSignal(str)
    all_parameters_changed: pyqtSignal = pyqtSignal()
    points_changed: pyqtSignal = pyqtSignal()
    varying_changed: pyqtSignal = pyqtSignal(str)
    constant_changed: pyqtSignal = pyqtSignal(float)
    parameters_saved: pyqtSignal = pyqtSignal()

    @staticmethod
    def direct(temp:float, field:float, a_dir:float, n_dir:float) -> float:
        """Direct part of model

        Args:
            temp (float): Temperature.
            field (float): Magnetic field strange.
            a_dir (float): 
            n_dir (float): 

        Returns:
            float: Direct
        """
        return a_dir * temp * power(field, n_dir)

    @staticmethod
    def qtm(field:float, b1:float, b2:float, b3:float) -> float:
        """QTM(Quantum) part of model

        Args:
            field (float): Magnetic field strange.
            b1 (float): 
            b2 (float): 
            b3 (float): 

        Returns:
            float: QTM
        """
        return b1*(1+b3*field*field)/(1+b2*field*field)

    @staticmethod
    def Raman(temp:float, c_raman:float, n_raman:float) -> float:
        """Raman part of model

        Args:
            temp (float): Temperature.
            c_raman (float): 
            n_raman (float): 

        Returns:
            float: Raman
        """
        return c_raman* power(temp, n_raman)

    @staticmethod
    def Orbach(temp:float, tau_0:float, delta_e:float):
        """Orbahc part of model

        Args:
            temp (float): Temperature.
            tau_0 (float): 
            delta_e (float): 

        Returns:
            float: Orbach
        """
        return tau_0 * exp(-delta_e/temp)

    @staticmethod
    def model(temp:float, field:float, a_dir:float, n_dir:float, b1:float, b2:float,
     b3: float, c_raman:float, n_raman:float, tau_0:float, delta_e:float) -> float:
        """Relaxation time model

        Args:
            temp (float): Temperature.
            field (float): Magnetic field strange.
            a_dir (float): 
            n_dir (float): 
            b1 (float): 
            b2 (float): 
            b3 (float): 
            c_raman (float): 
            n_raman (float): 
            tau_0 (float): 
            delta_e (float): 

        Returns:
            float: Inverse of relaxation time
        """
        return a_dir*temp*(field**n_dir) \
            + b1*(1+b3*field*field)/(1+b2*field*field) \
            + c_raman * power(temp, n_raman) \
            + tau_0 *exp(-delta_e/(temp))


    def __init__(self, name: str, compound: SettingsSource, collection):

        super().__init__()
        self._name: str = name
        self.residual_error: float = 0.0
        self.saved_residual_error: float = 0.0
        logaritmic = ["a_dir", "b1", "b2", "b3", "tau_0"]
        self.parameters: TauParameters  = tuple( 
            Parameter(name, compound.get_min(name), compound.get_max(name), is_log = name in logaritmic) for name in get_args(TAU_PARAMETER_NAME)
        )

        self.saved_parameters: TauParameters = tuple( 
            Parameter(name, compound.get_min(name), compound.get_max(name), is_log = name in logaritmic) for name in get_args(TAU_PARAMETER_NAME)
        )

        self._points: list[Point] = []

        self.varying: Literal["Field", "Temperature"] = "Field"
        self.constant: float = 0

        self._collection: Collection
        if collection is not None:
            self._collection = collection

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val:str):
        if len(val) < 1:
            raise ValueError("Tau fit name must be at least one character long")
        self._name = val
        self.name_changed.emit(val)

    def append_point(self, tau:float, temp:float, field:float, silent=False):
        """Append new point to TauFit.

        Args:
            tau (float): Relaxation time.
            temp (float): Temperature
            field (float): Magnetic field strength
            silent (bool, optional): Determined whether signal point_changed is emitted. Defaults to False.
        """
        point: Point = Point(tau, temp, field)
        self._points.append(point)
        if not silent:
            self.points_changed.emit()

    def get_hidden(self) -> tuple[list[float], list[float], list[float]]:
        """Get all hidden points

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength. 
        """
        r_tau: list[float] = []
        r_temp: list[float] = []
        r_field: list[float] = []
        for p in self._points:
            if p.is_hidden:
                r_tau.append(p.tau)
                r_temp.append(p.temp)
                r_field.append(p.field)

        return (r_tau, r_temp, r_field)


    def get_visible(self) -> tuple[list[float], list[float], list[float]]:
        """Get all visible points.

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength.
        """
        r_tau: list[float] = []
        r_temp: list[float] = []
        r_field: list[float] = []
        for p in self._points:
            if not p.is_hidden:
                r_tau.append(p.tau)
                r_temp.append(p.temp)
                r_field.append(p.field)

        return (r_tau, r_temp, r_field)

    def get_all(self) -> tuple[list[float], list[float], list[float]]:
        """Get all points.

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength.
        """
        tmp: list = list(zip(*[(p.tau, p.temp, p.field) for p in self._points]))
        return (tmp[0], tmp[1], tmp[2])

    def get_visible_s(self) -> tuple[list[float], list[float], list[float]]:
        """Get all visible points in actual slice.

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength.
        """
        r_tau: list[float] = []
        r_temp: list[float] = []
        r_field: list[float] = []
        for p in self._points:
            if not p.is_hidden:
                if self.get_constant_from_point(p) == self.constant:
                    r_tau.append(p.tau)
                    r_temp.append(p.temp)
                    r_field.append(p.field)
        return (r_tau, r_temp, r_field)

    def get_hidden_s(self) -> tuple[list[float], list[float], list[float]]:
        """Get all hidden points in actual slice.

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength.
        """
        r_tau: list[float] = []
        r_temp: list[float] = []
        r_field: list[float] = []
        for p in self._points:
            if p.is_hidden:
                if self.get_constant_from_point(p) == self.constant:
                    r_tau.append(p.tau)
                    r_temp.append(p.temp)
                    r_field.append(p.field)
        return (r_tau, r_temp, r_field)

    def get_all_s(self) -> tuple[list[float], list[float], list[float]]:
        """Get all points in actual slice.

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength.
        """
        tmp: list = list(zip(*[(p.tau, p.temp, p.field) for p in self._points if self.get_constant_from_point(p) == self.constant]))
        try:
            return (tmp[0], tmp[1], tmp[2])
        except Exception:
            return ([], [], [])

    def get_constant_from_point(self, p:Point) -> float:
        """Extravt value of actual constant ax from point.

        Args:
            p (Point): Actual point.

        Returns:
            float: Temperature or magnetic field strength depending on actual constant ax in slice.
        """
        if self.varying == "Field":
            return p.temp
        else:
            return p.field

    def set_varying(self, txt: Literal["Field","Temperature"]):
        """Set new varying ax in slice.

        Args:
            txt (Literal[&quot;Field&quot;,&quot;Temperature&quot;]): New varying ax in slice.
        """
        self.varying = txt
        self.varying_changed.emit(self.varying)

    def set_constant(self, value: float):
        """Set new constant ax in slice.

        Args:
            value (float): New constant ax in slice.
        """
        self.constant = value
        self.constant_changed.emit(value)

    def hide_point(self, v: float, z: float):
        """Hide point.

        Args:
            v (float): Value of actual varying ax of Point to hide.
            z (float): Relaxation time of Point to hide.
        """
        for p in self._points:
            if self.varying == "Field":
                if p.field == v and p.temp == self.constant and log(p.tau) == z:
                    p.is_hidden = not p.is_hidden
                    break
            else:
                if p.temp == 1/v and p.field == self.constant and log(p.tau) == z:
                    p.is_hidden = not p.is_hidden
                    break
        self.points_changed.emit()

    def delete_point(self, v: float, z: float):
        """Delete point

        Args:
            v (float): Value of actual varying ax of Point ro delete.
            z (float): Relaxation time of Point to delete.
        """
        old_points: list[Point] = self._points
        if self.varying == "Field":
            self._points = [p for p in self._points if (p.field, p.temp, log(p.tau)) != (v, self.constant, z)]
        else:
            self._points = [p for p in self._points if (p.temp,  p.field, log(p.tau)) != (1/v, self.constant, z)]

        if len(self._points) < 2:
            self._points = old_points

        self.points_changed.emit()

    def get_parameters_values(self) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """Get all parameters value.

        Returns:
            tuple[float, float, float, float, float, float, float, float, float, float]: Values of all parameters.
        """
        return tuple(p.value for p in self.parameters) # type: ignore

    def get_saved_parameters_values(self) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """Get all saved parameters value.

        Returns:
            tuple[float, float, float, float, float, float, float, float, float, float]: Values of all saved parameters.
        """
        return tuple(p.value for p in self.saved_parameters) # type: ignore

    def get_parameters_min_bounds(self) -> list[float]:
        """Get lower boundaries of all parameters.

        Returns:
            list[float]: Values of lower boundaries.
        """
        return [p.min for p in self.parameters]

    def get_parameters_max_bounds(self) -> list[float]:
        """Get upper boundaries of all parameters.

        Returns:
            list[float]: Values of upper boundaries.
        """
        return [p.max for p in self.parameters]

    def make_auto_fit(self, slice_flag=False):
        """Solve a nonlinear least-squares problem with bounds on the variables for cost_function().

        Args:
            slice_flag (bool, optional): Determines whether take into account only Points in actual slice. Defaults to False.
        """
        params = self.get_parameters_values()
        min: list[float] = self.get_parameters_min_bounds()
        max: list[float] = self.get_parameters_max_bounds()

        p:Parameter
        for i, p in enumerate(self.parameters):
            if p.is_blocked:
                min[i] = nextafter(p.value, min[i])
                max[i] = nextafter(p.value, max[i])

            if p.is_blocked_on_0:
                params = list(params)
                min[i] = nextafter(nextafter(0.0, 1), 0)
                max[i] = nextafter(nextafter(0.0, 1), 1)
                params[i] = nextafter(0.0, 1)

            if p.value < min[i] and not p.is_blocked_on_0:
                tmp: float = nextafter(min[i], max[i])
                p.set_value(tmp)
                params = list(params)
                params[i] = tmp

        bounds: tuple[list[float], list[float]] = (min, max)
        cost_f = partial(self.cost_function, slice=slice_flag)

        res = least_squares(cost_f, params, bounds=bounds)

        for i, p in enumerate(self.parameters):
            p.set_value(res.x[i])

        _, s, Vh = svd(res.jac, full_matrices=False)
        tol = finfo(float).eps*s[0]*np_max(res.jac.shape)
        w = s > tol
        cov = (Vh[w].T/s[w]**2) @ Vh[w]

        chi2dof = sum(res.fun**2)/(res.fun.size - res.x.size)
        cov *= chi2dof

        perr = sqrt(diag(cov))
        self.set_all_errors(res.cost, perr)

    def cost_function(self, p, slice=False):
        """Cost function minimalized in least_square method in fitting process.

        Args:
            p (_type_): All parameters.
            slice (bool, optional): Determines whether take into account only Points in actual slice. Defaults to False.

        Returns:
            _type_: Cost of fit.
        """
        if slice:
            tau, temp, field = self.get_all_s()
        else:
            tau, temp, field = self.get_all()
        temp = Series(temp)
        field = Series(field)
        tau = Series(tau)
        return power(log(TauFit.model(temp, field, *p)) - log(1/tau), 2)

    def set_all_errors(self, residual_error: float, params_error: list[float]):
        """Set all parameters error to new values.

        Args:
            residual_error (float): Residual error of least square problem.
            params_error (list[float]): Invidual parameters errors of least square problem.
        """
        self.residual_error = residual_error
        p: Parameter
        for i, p in enumerate(self.parameters):
            p.set_error(params_error[i], silent=True)
        self.all_parameters_changed.emit()

    def save(self):
        """Set all current parameters as saved.
        """
        for i, p in enumerate(self.parameters):
            s = self.saved_parameters[i]
            s.set_value(p.value)
            s.set_blocked(p.is_blocked)
            s.set_blocked_0(p.is_blocked_on_0)
            s.set_error(p.error)
        self.parameters_saved.emit()

    def reset(self):
        """Set all current parameters values to saved ones.
        """
        for i, p in enumerate(self.parameters):
            s = self.saved_parameters[i]
            p.set_value(s.value)
            p.set_blocked(s.is_blocked)
            p.set_blocked_0(s.is_blocked_on_0)
            p.set_error(s.error)

        self.all_parameters_changed.emit()

    def copy(self, other):
        """Set saved parameters of other as current parameters of self.

        Args:
            other (Relaxation): Relaxation to copy.
        """
        for i, p in enumerate(self.parameters):
            o = other.saved_parameters[i]
            p.set_value(o.value)
            p.set_blocked(o.is_blocked)
            p.set_blocked_0(o.is_blocked_on_0)
            p.set_error(o.error)

        self.all_parameters_changed.emit()

    def save_to_file(self):
        """Savig result to .csv file"""
        save_name, _ = QFileDialog.getSaveFileName(QWidget(), 'Save file')
        if save_name is not None:
            try:
                with open(save_name + ".csv", "w") as f:
                    self.get_result().to_csv(f.name, index=False, sep = ";")
            except Exception as e:
                print(e)
                return

    def get_result(self):
        """Get DataFrame with results of fitting process.

        Returns:
            DataFrame: Result of fit in DataFrame format.
        """
        df_param: DataFrame = DataFrame(columns=["Name", "Value", "Error"])
        p:Parameter
        for p in self.parameters:
            row = {"Name": p.name, "Value": p.value, "Error":p.error}
            df_param = df_param.append(row, ignore_index=True)
        
        tau, tmp, field = self.get_all()
        df_experimental:DataFrame = DataFrame(list(zip(tmp, field, tau)), columns=["T", "H", "tau"])

        x = linspace(min(tmp), max(tmp), 50)
        y = linspace(min(field), max(field), 50)
        X, Y = meshgrid(x,y)
        Z = 1/TauFit.model(X,Y, *self.get_saved_parameters_values())

        temp = []
        for x in X:
            for t in x:
                temp.append(t)

        fields = []
        for y in Y:
            for h in y:
                fields.append(h)

        tau = []
        for z in Z:
            for v in z:
                tau.append(v)

        df_model: DataFrame = DataFrame(list(zip(temp, fields, tau)), columns=["TempModel", "FieldModel", "TauModel"])

        all_temp = set()
        for t in tmp:
            all_temp.add(t)
        final_series_tmp = [Series()]*7
        for t in list(all_temp):
            field = linspace(min(field), max(field), 50)
            field = Series(field)
            tmp = Series([t] * 50)
            partial_result = self.partial_result(tmp, field , return_df=False)
            one_point_series = [tmp, field] + partial_result
            for s in range(len(final_series_tmp)):
               final_series_tmp[s] = final_series_tmp[s].append(one_point_series[s], ignore_index=True)

        df_tmp: DataFrame = DataFrame(list(zip(*final_series_tmp)), columns=["Temp", "Field", "Orbach", "Raman", "QTM", "Direct", "Tau"])

        final_series_field = [Series()]*7
        all_field = set()
        for f in field:
            all_field.add(f)

        for f in list(all_field):
            tmp = linspace(min(tmp), max(tmp), 50)
            tmp = Series(tmp)
            field = Series([f]*50)
            partial_result = self.partial_result(tmp, field, return_df=False)
            one_point_series = [tmp, field] + partial_result
            for s in range(len(final_series_field)):
                final_series_field[s] = final_series_field[s].append(one_point_series[s], ignore_index=True)

        df_field: DataFrame = DataFrame(list(zip(*final_series_field)), columns=["Temp", "Field", "Orbach", "Raman", "QTM", "Direct", "Tau"])
        return concat([df_param, df_experimental, df_model, df_tmp, df_field], axis=1)

    def partial_result(self, temp, field, return_df = True):
        p = self.get_saved_parameters_values()
        orbach= 1/TauFit.Orbach(temp, p[7], p[8])
        raman = 1/TauFit.Raman(temp, p[5], p[6])
        qtm = 1/TauFit.qtm(field, p[2], p[3], p[4])
        direct = 1/TauFit.direct(temp, field, p[0], p[1])
        sum = 1/TauFit.model(temp, field, *p)

        if return_df:
            return DataFrame(list(zip(orbach, raman, qtm, direct, sum)), columns=["Orbach Tau", "Raman Tau", "QTM Tau", "Direct Tau", "Tau"])
        else:
            return [orbach, raman, qtm, direct, sum]

    def get_jsonable(self) -> dict:
        """Marshal object to python dictionary.

        Returns:
            dict: Dictionary ready to save as .json
        """

        p_list: list[dict] = []
        for p in self.parameters:
            p_list.append(p.get_jsonable())

        s_p_list: list[dict] = []
        for p in self.saved_parameters:
            s_p_list.append(p.get_jsonable())

        jsonable = {
         "residual_error": self.residual_error , 
         "saved_residual_error": self.saved_residual_error,
         "parameters": p_list,
         "saved_parameters": s_p_list,
         "name": self._name,
         "constant": self.constant,
         "varying": self.varying,
         "points": [p.get_jsonable() for p in self._points]
        }

        return jsonable

    def update_from_json(self, f: dict):
        """From given dictionary recreate saved state of TauFit.

        Args:
            f (list[dict]): Result of self.get_jsonable()
        """
        self.residual_error = f["residual_error"]
        self.saved_residual_error = f["saved_residual_error"]
        self.constant = f["constant"]
        self.varying = f["varying"]
        for i, p in enumerate(self.parameters):
            p.update_from_json(f["parameters"][i])

        for j, s_p in enumerate(self.saved_parameters):
            s_p.update_from_json(f["saved_parameters"][j])

        for point in f["points"]:
            self.append_point(point["tau"], point["temp"], point["field"], point["is_hidden"])
