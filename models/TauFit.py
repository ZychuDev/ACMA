from PyQt6.QtCore import QObject, pyqtSignal, QLocale, QSize, Qt
from PyQt6.QtWidgets import QFileDialog, QWidget, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton, QMessageBox
from PyQt6.QtGui import QUndoStack, QUndoCommand,QDoubleValidator

from .Parameter import Parameter, TAU_PARAMETER_NAME
from .Point import Point

from protocols import SettingsSource, Collection

from readers import SettingsReader

from typing import get_args, Literal
from functools import partial

from scipy.optimize import least_squares # type: ignore
from scipy.linalg import svd #type: ignore

from numpy import power, exp, log, finfo, sqrt, diag, linspace, meshgrid, setdiff1d
from numpy import max as np_max

from math import nextafter, isclose

from pandas import Series, DataFrame, concat #type: ignore

TauParameters = tuple[Parameter, ...]

class TauFit(QObject):
    """Represent results of fitting process permorfmed on multiple Measurements.

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
    def Raman(temp:float, c_raman_1:float, n_raman_1:float) -> float:
        """Raman part of model

        Args:
            temp (float): Temperature.
            c_raman_1 (float): 
            n_raman_1 (float): 

        Returns:
            float: Raman
        """
        return c_raman_1* power(temp, n_raman_1)

    @staticmethod
    def Raman_2(temp:float, field:float, c_raman_2:float, n_raman_2:float, m_2: float) -> float:
        """Raman part of model

        Args:
            field(float): Field.
            temp (float): Temperature.
            c_raman_2 (float): 
            n_raman_2 (float): 
            m_2(float):

        Returns:
            float: Raman_2
        """
        return c_raman_2* power(temp, n_raman_2) * power(field, m_2)

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
    def V_d(temp:float, v: float, d: float):
        return d *(exp(v/temp)/pow(exp(v/temp) -1.0,2.0))
    
    @staticmethod
    def model(temp:float, field:float, a_dir:float, n_dir:float, b1:float, b2:float,
     b3: float, c_raman_1:float, n_raman_1:float, c_raman_2:float, n_raman_2:float, m_2:float, tau_0:float, delta_e:float, v:float, d:float) -> float:
        """Relaxation time model

        Args:
            temp (float): _description_
            field (float): _description_
            a_dir (float): _description_
            n_dir (float): _description_
            b1 (float): _description_
            b2 (float): _description_
            b3 (float): _description_
            c_raman_1 (float): _description_
            n_raman_1 (float): _description_
            c_raman_2 (float): _description_
            n_raman_2 (float): _description_
            m_2 (float): _description_
            tau_0 (float): _description_
            delta_e (float): _description_

        Returns:
            float: Inverse of relaxation time
        """

        return a_dir*temp*(field**n_dir) \
            + b1*(1+b3*field*field)/(1+b2*field*field) \
            + c_raman_1 * power(temp, n_raman_1) \
            + c_raman_2 * power(temp, n_raman_2) * power(field, m_2) \
            + tau_0 *exp(-delta_e/(temp)) \
            + d *(exp(v/temp)/pow(exp(v/temp) -1.0,2.0))


    def __init__(self, name: str, compound: SettingsSource, collection):

        super().__init__()
        self._name: str = name
        self._compound = compound
        self.residual_error: float = 0.0
        self.saved_residual_error: float = 0.0
        self.logaritmic = ["a_dir", "b1", "b2", "b3", "tau_0", "d"]

        self.parameters: TauParameters  = tuple( 
            Parameter(name, compound.get_min(name), compound.get_max(name), is_log = name in self.logaritmic) for name in get_args(TAU_PARAMETER_NAME)
        )

        self.saved_parameters: TauParameters = tuple( 
            Parameter(name, compound.get_min(name), compound.get_max(name), is_log = name in self.logaritmic) for name in get_args(TAU_PARAMETER_NAME)
        )

        self._points: list[Point] = []

        self.varying: Literal["Field", "Temperature"] = "Field"
        self.constant: float = 0

        self._collection: Collection["TauFit"]
        if collection is not None:
            self._collection = collection

        self._undo_stack: QUndoStack = QUndoStack()

        for p in self.parameters:
            p.reset_errors.connect(lambda: self.set_all_errors(0.0, [0.0] * len(get_args(TAU_PARAMETER_NAME)) ))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val:str):
        if len(val) < 1:
            raise ValueError("Tau fit name must be at least one character long")
        if self._collection is not None:
            self._collection.update_names(old_name=self._name, new_name=val)
        self._name = val
        self.name_changed.emit(val)

    class Rename(QUndoCommand):
        def __init__(self, tau_fit: "TauFit", new_name:str):
            super().__init__()
            self.tau_fit = tau_fit
            self.new_name = new_name
            self.old_name = tau_fit.name
            
        def redo(self) -> None:
            self.tau_fit.name = self.new_name

        def undo(self) -> None:
            self.tau_fit.name = self.old_name

    class HidePoint(QUndoCommand):
        def __init__(self, tau_fit: "TauFit", v: float, z: float):
            super().__init__()
            self.tau_fit: TauFit = tau_fit
            self.v: float = v
            self.z: float = z

        def redo(self) -> None:
            self.tau_fit._hide_point(self.v, self.z)

        def undo(self) -> None:
            self.tau_fit._hide_point(self.v, self.z)

    class DeletePoint(QUndoCommand):
        def __init__(self, tau_fit: "TauFit", v: float, z: float):
            super().__init__()
            self.tau_fit: TauFit = tau_fit
            self.v: float = v
            self.z: float = z
            self.point:Point = Point(1.0,1.0,1.0)

        def redo(self) -> None:
            self.point = self.tau_fit._delete_point(self.v, self.z)

        def undo(self) -> None:
            self.tau_fit._points.append(self.point)
            self.tau_fit.points_changed.emit()

    def set_name(self, new_name: str):
        self._undo_stack.push(self.Rename(self, new_name))

    def hide_point(self, v: float, z: float):
        """Hide point that can be undone.

        Args:
            v (float): Value of actual varying ax of Point to hide.
            z (float): Relaxation time of Point to hide.
        """
        self._undo_stack.push(self.HidePoint(self, v, z))

    def delete_point(self, v: float, z: float):
        """Delete point that can be undone.

        Args:
            v (float): Value of actual varying ax of Point ro delete.
            z (float): Relaxation time of Point to delete.
        """
        self._undo_stack.push(self.DeletePoint(self, v, z))

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

    def get_all_visible(self) -> tuple[list[float], list[float], list[float]]:
        """Get all points.

        Returns:
            tuple[list[float], list[float], list[float]]: Values in order: tau, temperature, magnetic field strength.
        """
        tmp: list = list(zip(*[(p.tau, p.temp, p.field) for p in self._points if not p.is_hidden]))
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

    def _hide_point(self, v: float, z: float):
        """Hide point.

        Args:
            v (float): Value of actual varying ax of Point to hide.
            z (float): Relaxation time of Point to hide.
        """
        
        for i, p in enumerate(self._points):
            if self.varying == "Field":
                if isclose(p.field, v, rel_tol=1e-09, abs_tol=0.0) and (abs(p.field-v) < abs(self._points[i-1].field-v) and abs(p.field-v) < abs(self._points[(i+1)%len(self._points)].field-v)) and p.temp == self.constant and log(p.tau) == z:
                    p.is_hidden = not p.is_hidden
                    break
            else:
                if isclose(p.temp, 1/v, rel_tol=1e-09, abs_tol=0.0)and (abs(p.temp-1/v) < abs(self._points[i-1].temp-1/v) and abs(p.temp-1/v) < abs(self._points[(i+1)%len(self._points)].field-v)) and p.field == self.constant and log(p.tau) == z:
                    p.is_hidden = not p.is_hidden
                    break
        self.points_changed.emit()

    def _delete_point(self, v: float, z: float):
        """Delete point

        Args:
            v (float): Value of actual varying ax of Point ro delete.
            z (float): Relaxation time of Point to delete.
        """

        tmp: list[Point]
        if self.varying == "Field":
            tmp = [p for p in self._points if (p.field, p.temp, log(p.tau)) != (v, self.constant, z)]

        else:
            tmp = [p for p in self._points if (not isclose(p.temp,1/v, rel_tol=1e-09, abs_tol=0.0) or p.field != self.constant or log(p.tau) != z)]
            

        if len(tmp) < 2:
            return

        if len(list(set(self._points) - set(tmp))) == 0:
            print("Error")
            raise Exception()
        
        point: Point = list(set(self._points) - set(tmp))[0]
    
        self._points = tmp

        self.points_changed.emit()
        return point



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
        blocked_parameters:list[Parameter] = []
        blocked_on_0_parameters:list[Parameter] = []
        not_blocked_parameters:list[Parameter] = []
        for p in self.parameters:
            if p.is_blocked_on_0:
                blocked_on_0_parameters.append(p)
            elif p.is_blocked:
                blocked_parameters.append(p)
            else:
                not_blocked_parameters.append(p)

        param_str:str = ""
        for p in not_blocked_parameters:
            param_str = param_str +f", {p.name}"

        

        not_blocked_names = [p.name for p in not_blocked_parameters]
        not_blocked_params = [p.value for p in not_blocked_parameters]
        blocked_on_0_parameters_names = [p.name for p in blocked_on_0_parameters]
        fit_bounds = ([p.min for p in not_blocked_parameters], [p.max for p in not_blocked_parameters])

        
        model_body = """return {a_dir}*temp*(field**{n_dir}) \
            + {b1}*(1+{b3}*field*field)/(1+{b2}*field*field) \
            + {c_raman_1} * power(temp, {n_raman_1}) \
            + {c_raman_2} * power(temp, {n_raman_2}) * power(field, {m_2}) \
            + {tau_0} *exp(-{delta_e}/(temp)) \
            + {d} *(exp({v}/temp)/pow(exp({v}/temp) -1.0,2.0)) """.format(
            a_dir = "a_dir" if ("a_dir" in not_blocked_names) else (0.0 if "a_dir" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "a_dir")).value),
            n_dir = "n_dir" if ("n_dir" in not_blocked_names) else (0.0 if "n_dir" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "n_dir")).value),
            b1 = "b1" if ("b1" in not_blocked_names) else (0.0 if "b1" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "b1")).value),
            b2 = "b2" if ("b2" in not_blocked_names) else (0.0 if "b2" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "b2")).value),
            b3 = "b3" if ("b3" in not_blocked_names) else (0.0 if "b3" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "b3")).value),
            c_raman_1 = "c_raman_1" if ("c_raman_1" in not_blocked_names) else (0.0 if "c_raman_1" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "c_raman_1")).value),
            n_raman_1 = "n_raman_1" if ("n_raman_1" in not_blocked_names) else (0.0 if "n_raman_1" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "n_raman_1")).value),
            c_raman_2 = "c_raman_2" if ("c_raman_2" in not_blocked_names) else (0.0 if "c_raman_2" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "c_raman_2")).value),
            n_raman_2 = "n_raman_2" if ("n_raman_2" in not_blocked_names) else (0.0 if "n_raman_2" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "n_raman_2")).value),
            m_2 = "m_2" if ("m_2" in not_blocked_names) else (0.0 if "m_2" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "m_2")).value),
            tau_0 = "tau_0" if ("tau_0" in not_blocked_names) else (0.0 if "tau_0" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "tau_0")).value),
            delta_e = "delta_e" if ("delta_e" in not_blocked_names) else (0.0 if "delta_e" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "delta_e")).value),
            v = "v" if ("v" in not_blocked_names) else (0.0 if "v" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "v")).value),
            d = "d" if ("d" in not_blocked_names) else (0.0 if "d" in blocked_on_0_parameters_names else next((x for x in blocked_parameters if x.name == "d")).value),
        )

        meta_model_str = """
def m_model(temp, field {param_str}):
    {model_body}
self.meta_model = m_model
            """.format(model_body= model_body, param_str=param_str)
        exec(meta_model_str, {"self":self, "power":power, "exp":exp})

        exec("""
def meta_auto_fit(self):
    def cost_function(p, slice=False):
        if slice:
            tau, temp, field = self.get_visible_s()
        else:
            tau, temp, field = self.get_all_visible()
        temp = Series(temp)
        field = Series(field)
        tau = Series(tau)
        return abs(log(self.meta_model(temp, field, *p)) - log(1/tau))

    settings: SettingsReader = SettingsReader()
    tole = settings.get_tolerances()
    try:
        try:
            res = least_squares(cost_function, params, bounds=bounds, ftol=tole["f_tol"], xtol=tole["x_tol"], gtol=tole["g_tol"])
        except ValueError as e:
            msg: QMessageBox = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("One of tolerances is to low. You can adjust them in settings.")
            msg.setText(str(e))
            msg.setWindowTitle("Auto fit failed")
            msg.exec()
            return
        _, s, Vh = svd(res.jac, full_matrices=False)
        tol = finfo(float).eps*s[0]*np_max(res.jac.shape)
        w = s > tol
        cov = (Vh[w].T/s[w]**2) @ Vh[w]

        chi2dof = sum(res.fun**2)/(res.fun.size - res.x.size)
        cov *= chi2dof

        perr = sqrt(diag(cov))
    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText("Something went wrong. Try change starting values of parameters.")
        msg.setText(str(e))
        msg.setWindowTitle("Auto fit failed")
        msg.exec()
        return

    for i, p in enumerate(not_blocked_parameters):
        p.set_value(res.x[i])
        p.set_error(perr[i], silent=True)
    for p in blocked_parameters:
        p.set_error(0.0, silent=True)
    for p in blocked_on_0_parameters:
        p.set_error(0.0, silent=True)
             
    self.residual_error = res.cost
    self.all_parameters_changed.emit()
    
meta_auto_fit(self)
        """, {
            "self":self, "slice":slice_flag, "Series": Series, "abs": abs, "log":log, "SettingsReader":SettingsReader, "QMessageBox":QMessageBox,
            "params":not_blocked_params, "bounds":fit_bounds, "svd":svd, "finfo":finfo, "np_max":np_max, "sum":sum, "sqrt":sqrt, "least_squares": least_squares, "diag":diag,
            "not_blocked_parameters":not_blocked_parameters, "blocked_parameters":blocked_parameters, "blocked_on_0_parameters":blocked_on_0_parameters,
        })

        return 

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
        return abs(log(TauFit.model(temp, field, *p)) - log(1/tau))

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
        self.saved_residual_error = self.residual_error
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

    def copy(self, other:"TauFit"):
        """Set saved parameters of other as current parameters of self.

        Args:
            other (Relaxation): Relaxation to copy.
        """
        for i, p in enumerate(self.parameters):
            o = other.saved_parameters[i]
            p.set_range(*o.get_range())
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
                with open(save_name + (".csv" if save_name[-4:] != ".csv" else ""), "w") as f:
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
            row = {"Name": p.name, "Value": p.value if not p.is_blocked_on_0 else 0.0, "Error":p.error}
            df_param = concat([df_param, DataFrame([row])], ignore_index=True)
        
        tau, tmp_o, field_o = self.get_all()
        df_experimental:DataFrame = DataFrame(list(zip(tmp_o, field_o, tau)), columns=["T", "H", "tau"])

        x = linspace(min(tmp_o), max(tmp_o), 50)
        y = linspace(min(field_o), max(field_o), 50)
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
        for t in tmp_o:
            all_temp.add(t)
        columns_names: list[str] = ["Temp", "Field", "Orbach", "Raman", "Raman_2", "QTM", "Direct", "V_d", "Tau"]
        final_series_tmp = [Series()]*len(columns_names)
        all_temp = list(all_temp)
        all_temp.sort()
        for t in all_temp:
            field = linspace(min(field_o), max(field_o), 50)
            field = Series(field)
            tmp = Series([t] * 50)
            partial_result = self.partial_result(tmp, field , return_df=False)
            one_point_series = [tmp, field] + partial_result
            for s in range(len(final_series_tmp)):
               final_series_tmp[s] = concat([final_series_tmp[s], one_point_series[s]], ignore_index=True)

        df_tmp: DataFrame = DataFrame(list(zip(*final_series_tmp)), columns=columns_names)
        final_series_field = [Series()]*(len(columns_names))
        all_field = set()
        for f in field_o:
            all_field.add(f)
        all_field = list(all_field)
        all_field.sort()
        for f in list(all_field):
            tmp = linspace(min(tmp_o), max(tmp_o), 50)
            tmp = Series(tmp)
            field = Series([f]*50)
            partial_result = self.partial_result(tmp, field, return_df=False)
            one_point_series = [tmp, field] + partial_result
            for s in range(len(final_series_field)):
                final_series_field[s] = concat([final_series_field[s], one_point_series[s]], ignore_index=True)
        df_field: DataFrame = DataFrame(list(zip(*final_series_field)), columns=["Temp", "Field", "Orbach", "Raman", "Raman_2", "QTM", "Direct", "V_d", "Tau"])
        return concat([df_param, df_experimental, df_model, df_tmp, df_field], axis=1)

    def partial_result(self, temp, field, return_df = True):
        p = self.get_saved_parameters_values()
        orbach= 1/TauFit.Orbach(temp, p[10], p[11])
        raman = 1/TauFit.Raman(temp, p[5], p[6])
        raman_2 = 1/TauFit.Raman_2(temp, field, p[7], p[8], p[9])
        qtm = 1/TauFit.qtm(field, p[2], p[3], p[4])
        direct = 1/TauFit.direct(temp, field, p[0], p[1])
        v_d = 1/TauFit.V_d(temp, p[12], p[13])
        sum = 1/TauFit.model(temp, field, *p)

        if return_df:
            return DataFrame(list(zip(orbach, raman, raman_2, qtm, direct, v_d, sum)), columns=["Orbach Tau", "Raman Tau", "Raman_2 Tau", "QTM Tau", "Direct Tau", "V_d Tau", "Tau"])
        else:
            return [orbach, raman, raman_2, qtm, direct, v_d, sum]

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

        name: TAU_PARAMETER_NAME = "v"
        tmp:dict 
        if not any(item["name"] == name for item in f['parameters']):
            tmp = Parameter(name, self._compound.get_min(name), self._compound.get_max(name), is_log = False).get_jsonable()
            f['parameters'].append(Parameter(name, self._compound.get_min(name), self._compound.get_max(name), is_log = False).get_jsonable())
            f['saved_parameters'].append(tmp)

        name = "d"
        if not any(item == name for item in f['parameters']):
            tmp = Parameter(name, self._compound.get_min(name), self._compound.get_max(name), is_log = name in self.logaritmic).get_jsonable()
            f['parameters'].append(tmp)
            f['saved_parameters'].append(tmp)

        for i, p in enumerate(self.parameters):
            p.update_from_json(f["parameters"][i])

        for j, s_p in enumerate(self.saved_parameters):
            s_p.update_from_json(f["saved_parameters"][j])

        for point in f["points"]:
            self.append_point(point["tau"], point["temp"], point["field"], point["is_hidden"])

    def adjust_parameters(self):
        dlg:QDialog = QDialog()
        default_locale: QLocale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        dlg.setLocale(default_locale)
        dlg.setWindowTitle("Set parameters ranges")
        layout: QVBoxLayout = QVBoxLayout()
        new_ranges: dict[str,tuple[QLineEdit, QLineEdit]] = {}
        p: str
        for p in self.parameters:
            l: QHBoxLayout = QHBoxLayout()

            v: QDoubleValidator = QDoubleValidator()
            loc: QLocale = QLocale(QLocale.c())
            loc.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
            v.setLocale(loc)

            low: QLineEdit = QLineEdit()
            low.setLocale(default_locale)
            low.setValidator(v)
            low.setText(str(p.min))

            up: QLineEdit = QLineEdit()
            up.setLocale(default_locale)
            up.setValidator(v)
            up.setText(str(p.max))

            label:QLabel = QLabel(p.symbol)
            label.setMinimumSize(QSize(65, 0))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            new_ranges[p.name] = (low, up)
            l.addWidget(low)
            l.addWidget(label)
            l.addWidget(up)

            layout.addLayout(l)

        buttons_layout: QHBoxLayout = QHBoxLayout()
        button: QPushButton = QPushButton("Apply")
        button.clicked.connect(lambda: self.set_new_ranges(new_ranges, False, dlg))

        force_button: QPushButton() = QPushButton("Force change")
        force_button.clicked.connect(lambda: self.set_new_ranges(new_ranges, True, dlg))

        buttons_layout.addWidget(button)
        buttons_layout.addWidget(force_button)
        layout.addLayout(buttons_layout)

        dlg.setLayout(layout)
        dlg.exec() 
    
    def set_new_ranges(self, edit_lines: dict[str,tuple[QLineEdit, QLineEdit]], force: bool, dlg:QDialog):
        pa: str
        message: str
        msg: QMessageBox
        for pa in edit_lines:
            if float(edit_lines[pa][0].text()) >=  float(edit_lines[pa][1].text()):
                message = f"Each upper bound must be strictly greater than lower bound\n Bounds for parameter: {p} are invalid"
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setText(message)
                msg.setWindowTitle("Incorrect bounds")
                msg.exec()
                return

        p: Parameter
        pp: Parameter
        t_f: TauFit #self

        lower: float
        upper: float
        value: float

        if not force:
            for p in self.saved_parameters:
                lower = float(edit_lines[p.name][0].text())
                upper = float(edit_lines[p.name][1].text())
                value = p.get_value()
                if value < lower or value > upper:
                    message = (f"Bounds for parameter: {p.symbol} are invalid\n"
                    +f"In {self.name} saved value of paramater is not in given range\n"
                    +f"Lower : {lower}\nUpper: {upper}\nActual: {value}\n" 
                    +"Force new ranges or change saved values of the parameters")

                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setText(message)
                    msg.setWindowTitle("Incorrect bounds")
                    msg.exec()
                    return
        
        for j in range(len(self.saved_parameters)):
            p = self.saved_parameters[j]
            lower = float(edit_lines[p.name][0].text())
            upper = float(edit_lines[p.name][1].text())
            value = p.get_value()
            error = p.error
            p.set_range(lower, upper)
            if value < lower:
                value = lower
            if value > upper:
                value = upper
            p.set_value(value, new_error=error)
            self.residual_error = 0.0
            for k in range(len(self.parameters)):
                pp = self.parameters[k]
                if pp.name == p.name:
                    value = pp.get_value()
                    value = min(upper, max(value, lower))
                    pp.set_range(lower, upper)
                    pp.set_value(value)

        dlg.close()

    def undo(self):
        self._undo_stack.undo()

    def redo(self):
        self._undo_stack.redo()
