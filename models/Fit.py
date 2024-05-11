from PyQt6.QtCore import QObject, pyqtSignal, pyqtBoundSignal, QLocale, QSize, Qt
from PyQt6.QtWidgets import QFileDialog, QWidget, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton
from PyQt6.QtGui import QUndoStack, QUndoCommand, QDoubleValidator

from .Relaxation import Relaxation
from .Measurement import Measurement 
from .Parameter import Parameter

from protocols import SettingsSource, Collection

from readers import SettingsReader

from pandas import DataFrame, Series, concat # type: ignore
from numpy import ndarray, pi, power, finfo, diag, sum, sqrt, pi, power, linspace, log10, logspace
from numpy import max as np_max
from numpy import min as np_min
from scipy.optimize import least_squares # type: ignore
from scipy.linalg import svd #type: ignore
from math import nextafter

from typing import Self 

class Fit(QObject):
    """Represent one fit for Havriliak-Negami model with n-relaxation.

        Args:
            name (str): Name of fit.
            df (DataFrame): Measurement data(processed data from magnetometer).
            temp (float): Temperature measured during the measurement.
            field (float): Magnetic field strength during measurement.
            compound (SettingsSource): Examined compound.
            collection (Collection | None): The collection to which it belongs.

        Attributes:
            name_changed: Emitted when name change. Contains new name.
            df_changed: Emitted when at least one row in df changed.
            df_point_deleted: Emitted when row in df is removed.
            deletion_imposible: Emitted when deletion operation could not be performed.
    """

    name_changed: pyqtSignal = pyqtSignal(str)
    df_changed: pyqtSignal = pyqtSignal()
    df_point_deleted: pyqtSignal = pyqtSignal()
    deletion_imposible:pyqtSignal = pyqtSignal()

    @staticmethod
    def model(logFrequency: ndarray, alpha: float, beta: float, tau: float, chi_dif: float, chi_s: float) -> ndarray:
        """Implemntation of Havriliak-Negami model

        Args:
            logFrequency (ndarray): Logarytm of frequency.
            alpha (float): Alpha parameter
            beta (float): Beta parameter
            tau (float): Time of relaxation.
            chi_dif (float): chi_t - chi_s
            chi_s (float): 

        Returns:
            ndarray: Model predictions for each point in domain
        """
        return chi_s + (chi_dif)/((1 + (10**logFrequency*2*pi * power(10, tau) * 1j )**(1- alpha))**beta)

    @staticmethod
    def from_measurement(measurement: Measurement, compound:SettingsSource, nr_of_relaxations: int = 1):
        """ Create new Fit from Measurement and append it to the collection.


        Args:
            measurement (Measurement): Measurement from which Fit will be created.
            compound (SettingsSource): Source of boundaries for parameters.
            nr_of_relaxations (int, optional): Number of relaxations in Havriliak-Negami model. Defaults to 1.

        Returns:
            Fit: Created Fit
        """
        df = measurement._df
        if df.isnull().values.any():
            msg: QMessageBox = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(f"Some data in measurement named: {measurement._name} is missing.\n Fit was not created!")
            msg.setWindowTitle("Fit creation canceled")
            msg.exec()
            return None
        
        fit_name: str = measurement._name + "_Fit_Frequency"
        fit: Fit =  Fit(fit_name, measurement._df.copy(), measurement._tmp, measurement._field, compound, None)

        fit.relaxations = []
        i: int
        for i in range(nr_of_relaxations):
            fit.relaxations.append(Relaxation(compound))

        return fit
    def __init__(self, name: str, df: DataFrame, temp: float, field: float, compound:SettingsSource, collection: Collection["Fit"]|None):
        super().__init__()
        self._name: str = name
        self._df: DataFrame = df

        self._tmp: float = temp
        self._field: float = field

        self.relaxations: list[Relaxation] = []

        self._compound: SettingsSource = compound
        self._collection: Collection["Fit"] | None
        if collection is not None:
            self._collection = collection

        self._undo_stack: QUndoStack = QUndoStack()
        self.resolution = 50 # TO::DO 

        for r in self.relaxations:
            r.reset_errors.connect(self.reset_errors_in_all_relaxations)

    def reset_errors_in_all_relaxations(self):
        for r in self.relaxations:
            r.set_all_errors(0.0, [0.0,0.0,0.0,0.0,0.0])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val:str):
        if len(val) < 1:
            raise ValueError("Compund name must be at least one character long")
        if self._collection is not None:
            self._collection.update_names(old_name=self._name, new_name=val)
        self._name = val
        self.name_changed.emit(val)

    class Rename(QUndoCommand):
        def __init__(self, fit: "Fit", new_name:str):
            super().__init__()
            self._fit = fit
            self.new_name = new_name
            self.old_name = fit.name
            
        def redo(self) -> None:
            self._fit.name = self.new_name

        def undo(self) -> None:
            self._fit.name = self.old_name

    class HidePoint(QUndoCommand):
        def __init__(self, fit: "Fit", x: float, x_str: str):
            super().__init__()
            self.fit: Fit = fit
            self.x: float = x
            self.x_str: str = x_str

        def redo(self) -> None:
            self.fit._hide_point(self.x, self.x_str)

        def undo(self) -> None:
            self.fit._hide_point(self.x, self.x_str)

    class DeletePoint(QUndoCommand):
        def __init__(self, fit: "Fit", x: float, x_str: str, error_signal: pyqtBoundSignal):

            super().__init__()
            self.fit: Fit = fit
            self.x: float = x
            self.x_str: str = x_str
            self.point: DataFrame = DataFrame()
            self.error_signal = error_signal

        def redo(self) -> None:
            try:
                self.point = self.fit._delete_point(self.x, self.x_str)
            except IndexError:
                self.error_signal.emit()

        def undo(self) -> None:
            self.fit._df = concat([self.fit._df, self.point])
            self.fit.df_changed.emit()

    @property
    def molar_mass(self):
        return self._molar_mass

    @molar_mass.setter
    def molar_mass(self, val:float):
        if val <= 0:
            raise ValueError("Molar mass must be greater than 0")

        self._molar_mass = val

    def set_name(self, new_name: str):
        """Sets fit name. This action can be undone.

        Args:
            new_name (str): New measurement name
        """
        self._undo_stack.push(self.Rename(self, new_name))
    
    def hide_point(self, x: float, x_str: str):
        """Change point visibility on the opposite of actual. This action can be undone.

        Args:
            x (float): Value of point for domain column.
            x_str (str): Name of domain column.
        """
        self._undo_stack.push(self.HidePoint(self, x, x_str))

    def delete_point(self, x: float, x_str: str):
        """Delete point. This action can be undone.

        Args:
            x (float): Value of point for domain column.
            x_str (str): Name of domain column.

        Raises:
            IndexError: Raised when there is not enough points to delete any more.
        """
        self._undo_stack.push(self.DeletePoint(self, x, x_str, self.deletion_imposible))

    def _hide_point(self, x: float, x_str: str):
        """Hide point 

        Args:
            x (float): Value of point for domain column.
            x_str (str): Name of domain column.
        """
        actual: bool = bool(self._df.loc[self._df[x_str] == x]['Hidden'].values[0])
        self._df.loc[self._df[x_str] == x, "Hidden"] = not actual
        self.df_changed.emit()

    def _delete_point(self, x: float, x_str: str):
        """Delete point 

        Args:
            x (float): Value of point for domain column.
            x_str (str): Name of domain column.
        """
        if self._df.shape[0] == 2:
            raise IndexError

        point: DataFrame = self._df.loc[self._df[x_str] == x]
        self._df.drop(self._df.loc[self._df[x_str] == x].index, inplace=True)
        self.df_point_deleted.emit()
        return point


    def cost_function(self, p):
        """Cost function minimalized in least_square method in fitting process.

        Args:
            p (_type_): Parameters for all relaxations.

        Returns:
            _type_: Cost of fit.
        """
        rest = self._df.loc[self._df["Hidden"] == False]

        sum_real = 0
        sum_img = 0
        i = 0
        while i < len(self.relaxations):
            r = Fit.model(rest["FrequencyLog"].values, p[0+i*5], p[1+i*5], p[2+i*5], p[3+i*5], p[4+i*5])
            sum_real += r.real
            sum_img += -r.imag

            i += 1

        dif_real = abs((sum_real - rest['ChiPrimeMol']))
        dif_img = abs((sum_img - rest['ChiBisMol']))


        return  dif_real + dif_img

    def make_auto_fit(self, auto: bool = False, next_fit: Self = None): # type: ignore
        """Solve a nonlinear least-squares problem with bounds on the variables for cost_function().

        Args:
            auto (bool, optional): Determines whether fit was explicitly call by user. Defaults to False.
            next_fit (Self, optional): Fit to transfer parameters value in case of performing automated fit process for mutiple Fits. Defaults to None.
        """
        if len(self.relaxations) == 1:
            not_blocked_parameters:list[Parameter] = []
            blocked_parameters:list[Parameter] = []  
            fit_bounds = ([p.min for p in not_blocked_parameters], [p.max for p in not_blocked_parameters])
            
            for j, p in enumerate(self.relaxations[0].parameters):
                if p.is_blocked:
                    blocked_parameters.append(p)
                else:
                    not_blocked_parameters.append(p)

            param_str = ""
            for p in not_blocked_parameters:
                param_str = param_str +f", {p.name}"

            not_blocked_names = [p.name for p in not_blocked_parameters]
            not_blocked_params = [p.value for p in not_blocked_parameters]

            model_body = "return {chi_s} + ({chi_dif})/((1 + (10**logFrequency*2*pi * power(10, {tau}) * 1j )**(1- {alpha}))**{beta})".format(
                chi_s="chi_s" if "chi_s" in not_blocked_names else next((x for x in blocked_parameters if x.name == "chi_s")).value,
                chi_dif="chi_dif" if "chi_dif" in not_blocked_names else next((x for x in blocked_parameters if x.name == "chi_dif")).value,
                tau="log10_tau" if "log10_tau" in not_blocked_names else next((x for x in blocked_parameters if x.name == "log10_tau")).value,
                alpha="alpha" if "alpha" in not_blocked_names else next((x for x in blocked_parameters if x.name == "alpha")).value,
                beta="beta" if "beta" in not_blocked_names else next((x for x in blocked_parameters if x.name == "beta")).value,
                )
            meta_model_str = """
def m_model(logFrequency {param_str}):
    {model_body}
self.meta_model = m_model
            """.format(model_body= model_body, param_str=param_str)

            exec(meta_model_str, {"self":self, "pi":pi, "power":power})
            exec("""
def meta_auto_fit(self):
    def cost_function(p):
        rest = self._df.loc[self._df["Hidden"] == False]
        sum_real = 0
        sum_img = 0

        r = self.meta_model(rest["FrequencyLog"].values, *p)
        sum_real += r.real
        sum_img += -r.imag

        dif_real = abs((sum_real - rest['ChiPrimeMol']))
        dif_img = abs((sum_img - rest['ChiBisMol']))
                 
        return  dif_real + dif_img
                 
    settings: SettingsReader = SettingsReader()
    tole = settings.get_tolerances()

    try:
        try:
            res = least_squares(cost_function, not_blocked_params, bounds=bounds, ftol=tole["f_tol"], xtol=tole["x_tol"], gtol=tole["g_tol"])
        except ValueError as e:
            msg: QMessageBox = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("One of tolerances is to low. You can adjust them in settings.")
            msg.setText(str(e))
            msg.setWindowTitle("Auto fit failed")
            msg.exec()
            return
        _, s, Vh = svd(res.jac, full_matrices=False)
        tol = finfo(float).eps * s[0] * np_max(res.jac.shape)
        w = s > tol
        cov = (Vh[w].T/s[w]**2) @ Vh[w] # robust covariance matrix

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

    for i,p in enumerate(not_blocked_parameters):
        p.set_value(res.x[i])
        p.set_error(perr[i], silent=True)
                 
    for p in blocked_parameters:
        p.set_error(0.0, silent=True)
                 
    self.relaxations[0].residual_error = res.cost
    self.relaxations[0].all_parameters_changed.emit()  

                   
    if auto:
        self.save_all_relaxations()
        if next_fit != None: #type: ignore
            self.copy_all_relxations(next_fit)   
                              
meta_auto_fit(self)
                    """, {"self": self, "SettingsReader":SettingsReader, "not_blocked_params": not_blocked_params, "bounds":fit_bounds, "QMessageBox":QMessageBox, "svd":svd,
                          "finfo":finfo, "np_max":np_max, "sum":sum, "sqrt":sqrt, "not_blocked_parameters":not_blocked_parameters, "blocked_parameters": blocked_parameters,
                           "next_fit": next_fit, "least_squares": least_squares, "diag":diag, "auto":auto })
            return
        else:
            params: tuple = ()
            min: list = []
            max: list = []
            for r_nr in range(len(self.relaxations)):
                relaxation = self.relaxations[r_nr]
                params = params + relaxation.get_parameters_values()
                min = min + relaxation.get_parameters_min_bounds()
                max = max + relaxation.get_parameters_max_bounds()

            
            r: Relaxation
            i: int
            for i, r in enumerate(self.relaxations):
                for j, p in enumerate(r.parameters):
                    if p.is_blocked:
                        min[j + i*len(r.parameters)] = nextafter(p.value, min[j + i*len(r.parameters)])
                        max[j + i*len(r.parameters)] = nextafter(p.value, max[j + i*len(r.parameters)])

            bounds: tuple[list[float], list[float]] = (min, max)
            minimal: float = np_min(params)
            maximal: float = np_max(params)

            settings: SettingsReader = SettingsReader()
            tole = settings.get_tolerances()
            try:
                try:
                    res = least_squares(self.cost_function, params, bounds=bounds, ftol=tole["f_tol"], xtol=tole["x_tol"], gtol=tole["g_tol"])
                except ValueError as e:
                    msg: QMessageBox = QMessageBox()
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setText("One of tolerances is to low. You can adjust them in settings.\n")
                    msg.setText(str(e))
                    msg.setWindowTitle("Auto fit failed")
                    msg.exec()
                    return

                _, s, Vh = svd(res.jac, full_matrices=False)
                tol = finfo(float).eps * s[0] * np_max(res.jac.shape)
                w = s > tol
                cov = (Vh[w].T/s[w]**2) @ Vh[w] # robust covariance matrix

                chi2dof = sum(res.fun**2)/(res.fun.size - res.x.size)
                cov *= chi2dof

                perr = sqrt(diag(cov))
            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setText("Something went wrong. Try change starting values of parameters.\n")
                msg.setText(str(e))
                msg.setWindowTitle("Auto fit failed")
                msg.exec()
                return

            for i, r in enumerate(self.relaxations):
                for j, p in enumerate(r.parameters):
                    p.set_value(res.x[j + i*len(r.parameters)])

            for i, r in enumerate(self.relaxations):
                r.set_all_errors(res.cost, perr[i*5 : i*5+5])

            if auto:
                self.save_all_relaxations()
                if next_fit != None: #type: ignore
                    self.copy_all_relxations(next_fit)

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

    def get_result(self,) -> DataFrame:
        """Get DataFrame with results of fitting process.

        Returns:
            DataFrame: Result of fit in DataFrame format.
        """

        df_param: DataFrame = DataFrame([['T', self._tmp, 0], ['H', self._field, 0]], columns=['Name', 'Value','Error'])

    
        df_experimental: DataFrame = self._df[["Frequency", "ChiPrimeMol","ChiBisMol"]]
        columns_names: list[str] = [f"Frequency T={self._tmp} H={self._field}",
            f"ChiPrimeMol T={self._tmp} H={self._field}", f"ChiBisMol T={self._tmp} H={self._field}"]
        df_experimental.columns = columns_names
        df_experimental.reset_index(drop=True, inplace=True)
        
        df_model_final: DataFrame = DataFrame()
        for i, r in enumerate(self.relaxations):
            for p in r.saved_parameters:
                name:str = p.name if p.name != "chi_dif" else "chi_t-chi_s"
                row = { "Name": f"{name}{i+1}", "Value": p.value if not p.is_blocked_on_0 else 0.0, "Error": p.error}
                df_param = concat([df_param, DataFrame([row])], ignore_index=True)


            df_model: DataFrame = DataFrame()
            displayed: DataFrame = self._df.loc[self._df["Hidden"] == False]

            xx:ndarray = logspace(log10(displayed["Frequency"].min()), log10(displayed["Frequency"].max()), self.resolution)
            df_model["Model"+columns_names[0]] = Series(xx)
            yy = Fit.model(log10(xx), *self.relaxations[i].get_saved_parameters_values())
            df_model["Model"+columns_names[1]] = Series(yy.real)
            df_model["Model"+columns_names[2]] = Series(-yy.imag)
            df_model_final = concat([df_model_final, df_model], axis=1)

        if i != 0:
            columns = list(df_model_final.columns)
            for j in range(0, len(columns), 3):
                rel_str:str = f" rel_nr={j//3 + 1}"
                columns[j] += rel_str
                columns[j+1] += rel_str
                columns[j+2] += rel_str
            df_model_final.columns = columns

        df: DataFrame = concat([df_param, df_experimental, df_model_final], axis=1)
        return df

    def save_all_relaxations(self):
        """Save current parameters of all relaxations.
        """
        for r in self.relaxations:
            r.save()

    def copy_all_relxations(self, other: Self): #type: ignore
        """Copy saved parameters for all relaxations"""
        for i, r in enumerate(other.relaxations): #type: ignore
                r.copy(self.relaxations[i])

    def get_jsonable(self) -> dict:
        """Marshal object to python dictionary.

        Returns:
            dict: Dictionary ready to save as .json
        """

        r_list: list[dict] = []
        for r in self.relaxations:
            r_list.append(r.get_jsonable())
        jsonable = {
         "name": self._name, 
         "df": self._df.to_json(),
         "tmp": self._tmp,
         "field": self._field,
         "relaxations": r_list
        }
        return jsonable

    def update_relaxations_from_json(self, relaxations_json: list[dict]):
        """From given dictionary recreate saved state of relaxations.

        Args:
            relaxations_json (list[dict]): Result of self.get_jsonable()
        """
        self.relaxations = []
        for i, r_j in enumerate(relaxations_json):
            r: Relaxation = Relaxation(self._compound)
            r.residual_error = r_j["residual_error"]
            r.saved_residual_error = r_j["saved_residual_error"]
            r.was_saved = r_j["was_saved"]
            
            for j, p in enumerate(r.parameters):
                p.update_from_json(r_j["parameters"][j])

            for k, s_p in enumerate(r.saved_parameters):
                s_p.update_from_json(r_j["saved_parameters"][k])

            self.relaxations.append(r)

    def adjust_range(self, r_nr: int):
        dlg:QDialog = QDialog()
        default_locale: QLocale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        dlg.setLocale(default_locale)
        dlg.setWindowTitle(f"Set parameters ranges for relaxation nr {r_nr+1}")
        layout: QVBoxLayout = QVBoxLayout()
        new_ranges: dict[str,tuple[QLineEdit, QLineEdit]] = {}
        p: str

        for p in self.relaxations[r_nr].parameters:
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
        button.clicked.connect(lambda: self.set_new_ranges(new_ranges, False, dlg, r_nr))

        force_button: QPushButton() = QPushButton("Force change")
        force_button.clicked.connect(lambda: self.set_new_ranges(new_ranges, True, dlg, r_nr))

        buttons_layout.addWidget(button)
        buttons_layout.addWidget(force_button)
        layout.addLayout(buttons_layout)

        guess_button = QPushButton("Guess the magnetic susceptibility parameters")
        guess_button.clicked.connect(lambda: self.guess(new_ranges))
        layout.addWidget(guess_button)
        dlg.setLayout(layout)
        dlg.exec()


    def guess(self, ranges: dict[str,tuple[QLineEdit, QLineEdit]]):
        max_chi_prime: float = -1e16
        max_chi_bis: float = -1e16

        max_chi_prime = max(self._df["ChiPrimeMol"].max(), max_chi_prime)
        max_chi_bis = max(self._df["ChiBisMol"].max(), max_chi_bis)


        ranges["chi_dif"][0].setText("0.0")
        ranges["chi_dif"][1].setText(f"{5.0*max_chi_bis}")
        ranges["chi_s"][0].setText("0.0")
        ranges["chi_s"][1].setText(f"{max_chi_prime}")

    def set_new_ranges(self, edit_lines: dict[str,tuple[QLineEdit, QLineEdit]], force: bool, dlg:QDialog, r_nr: int, close=True):
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

        f: Fit
        r: Relaxation
        p: Parameter
        pp: Parameter

        lower: float
        upper: float
        value: float

        if not force:
            r = self.relaxations[r_nr]
            for p in r.saved_parameters:
                lower = float(edit_lines[p.name][0].text())
                upper = float(edit_lines[p.name][1].text())
                value = p.get_value()
                if value < lower or value > upper:
                    message = (f"Bounds for parameter: {p.name} are invalid\n"
                    +f"In {f.name} saved value of paramater is not in given range\n"
                    +f"Lower : {lower}\nUpper: {upper}\nActual: {value}\n" 
                    +"Force new ranges or change saved values of the parameters")
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setText(message)
                    msg.setWindowTitle("Incorrect bounds")
                    msg.exec()
                    return
        
        r = self.relaxations[r_nr]
        for k in range(len(r.saved_parameters)):
            p = r.saved_parameters[k]
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
            r.residual_error = 0.0
            for m in range(len(r.parameters)):
                pp = r.parameters[m]
                if pp.name == p.name:
                    value = pp.get_value()
                    value = min(upper, max(value, lower))
                    pp.set_range(lower, upper)
                    pp.set_value(value)
                    
        if close:       
            dlg.close()

    def undo(self):
        self._undo_stack.undo()

    def redo(self):
        self._undo_stack.redo()