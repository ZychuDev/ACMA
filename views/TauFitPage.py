from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout,
 QPushButton, QTableWidget, QAbstractScrollArea, QCheckBox,
 QAbstractItemView, QTableWidgetItem, QDialog, QComboBox, QSplitter)
from PyQt6.QtCore import QSize, QMetaObject, QObject, Qt
from PyQt6.QtGui import QFont, QPalette, QBrush, QColor, QKeySequence, QShortcut

from models import TauFit, TAU_PARAMETER_NAME
from .ParameterSlider import ParameterSlider

from matplotlib.figure import Figure # type: ignore
from matplotlib.pyplot import figure, Axes # type: ignore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg # type: ignore
from matplotlib.colors import CSS4_COLORS # type: ignore
from matplotlib.ticker import LinearLocator # type: ignore
import matplotlib.colors as mcolors # type: ignore

from typing import get_args, cast, Literal

from time import time

from numpy import log, meshgrid, ones, linspace, isnan, isinf, concatenate
from pandas import Series # type: ignore

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: int=10, height: int=5, dpi: int=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.ax: Axes = fig.add_subplot()
        self.ax.grid(True, linestyle='--', linewidth=1, color=CSS4_COLORS["silver"])
        self.ax.xaxis.set_major_locator(LinearLocator(10))
        self.ax.yaxis.set_major_locator(LinearLocator(8))
        self.temp_label = r"$\frac{K}{T}$"
        self.field_label = r"$\frac{H}{Oe}$"
        title = r"$\tau^{-1}=A_{dir}TH^{N_{dir}} + \frac{B_1(1+B_3H^2)}{1+B_2H^2} + C_{Raman_{1}}T^{N_{Raman_{1}}} + C_{Raman_{2}}T^{N_{Raman_{2}}}H^{m_{2}}+\tau_0^{-1}\exp{\frac{-\Delta E}{T}} + D\frac{\exp{\frac{V}{T}}}{(\exp{\frac{V}{T}}-1)^2}$"
        self.ax.set(xlabel=self.temp_label, ylabel=r"$\ln{\frac{\tau}{s}}$",
         title=title)


class MplCanvas3D(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: int=10, height: int=8, dpi: int=100):
        self.fig: Figure = figure(figsize=(1,1), dpi = int(100), constrained_layout=True)
        self.fig.patch.set_facecolor("#ffffff")
        super().__init__(self.fig)
        self.axes: Axes = self.fig.add_subplot(projection='3d')
        self.axes.set_facecolor("#ffffff")

        self.axes.set_xlabel(r'$\frac{1}{T}$', rotation = 0, fontsize=15)
        self.axes.set_ylabel(r'$\frac{H}{Oe}$', rotation = 0, fontsize=15)
        self.axes.set_zlabel(r'$\ln{\frac{\tau}{s}}$', rotation = 0, fontsize=15)

        self.axes.xaxis.set_rotate_label(False) 
        self.axes.yaxis.set_rotate_label(False) 
        self.axes.zaxis.set_rotate_label(False) 



class TauFitPage(QWidget):
    def __init__(self):
        super().__init__()
        self.tau_fit: TauFit = None
        self.resolution = 50
        self.sliders: list[ParameterSlider] = []

        horizontal_layout: QHBoxLayout = QHBoxLayout()
        horizontal_spliter: QSplitter = QSplitter()

        p: QPalette = self.palette()
        p.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(p)
        self.setAutoFillBackground(True)

        left_layout: QVBoxLayout = QVBoxLayout()

        self.title_label: QLabel = QLabel("Title")
        self.title_label.setMaximumSize(QSize(16777215, 45))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font:QFont = QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(100)
        self.title_label.setFont(font)
        left_layout.addWidget(self.title_label, stretch=1)

        for parameter in get_args(TAU_PARAMETER_NAME):
            ps: ParameterSlider = ParameterSlider()
            self.sliders.append(ps)
            left_layout.addWidget(ps)

        control_layout: QHBoxLayout = QHBoxLayout()

        col_1: QVBoxLayout = QVBoxLayout()
        self.adjust_range_button: QPushButton = QPushButton("Adjust ranges")
        self.copy_parameters_button: QPushButton = QPushButton("Copy parameters")
        self.scale_checkbox: QCheckBox = QCheckBox("Scale slice for model")
        col_1.addWidget(self.adjust_range_button)
        col_1.addWidget(self.copy_parameters_button)
        col_1.addWidget(self.scale_checkbox)
        control_layout.addLayout(col_1, stretch=1)

        col_2: QVBoxLayout = QVBoxLayout()
        self.fit_button: QPushButton = QPushButton("Fit for all")
        self.save_button: QPushButton = QPushButton("Save")
        self.reset_button: QPushButton = QPushButton("Reset parameters")
        col_2.addWidget(self.fit_button)
        col_2.addWidget(self.save_button)
        col_2.addWidget(self.reset_button)
        control_layout.addLayout(col_2, stretch=1)

        col_3: QVBoxLayout = QVBoxLayout()
        self.fit_for_slice_button: QPushButton = QPushButton("Fit for slice")
        self.varying_slice_combo_box: QComboBox = QComboBox()
        self.varying_slice_combo_box.addItems(["Field", "Temperature"])
        self.constant_slice_label: QLabel = QLabel("Temperature")
        self.constant_value_slice: QComboBox = QComboBox()
        constant_slice_layout: QHBoxLayout = QHBoxLayout()
        constant_slice_layout.addWidget(self.constant_slice_label)
        constant_slice_layout.addWidget(self.constant_value_slice)
        col_3.addWidget(self.fit_for_slice_button)
        col_3.addWidget(self.varying_slice_combo_box)
        col_3.addLayout(constant_slice_layout)
        control_layout.addLayout(col_3, stretch=1)

        left_layout.addLayout(control_layout)

        palette: QPalette = QPalette()
        brush = QBrush(QColor(255, 244, 244))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Base, brush)
        brush = QBrush(QColor(255, 244, 244))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Base, brush)
        brush = QBrush(QColor(240, 240, 240))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, brush)

        fit_error: QTableWidget = QTableWidget()
        fit_error.setPalette(palette)
        fit_error.setAutoFillBackground(True)
        fit_error.setMinimumSize(QSize(255, 100))
        fit_error.setMaximumSize(QSize(16777215, 100))
        fit_error.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        fit_error.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        fit_error.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        fit_error.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        fit_error.setDragEnabled(False)
        fit_error.setAlternatingRowColors(False)
        fit_error.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        fit_error.setColumnCount(len(get_args(TAU_PARAMETER_NAME))+1)
        fit_error.setRowCount(2)

        fit_error.setVerticalHeaderItem(0, QTableWidgetItem("Saved: "))
        fit_error.setVerticalHeaderItem(1, QTableWidgetItem("Current: "))

        header = fit_error.verticalHeader()
        header.setVisible(True)
        header.setDefaultSectionSize(50)

        header = fit_error.horizontalHeader()
        header.setCascadingSectionResizes(True)
        header.setDefaultSectionSize(210)
        header.setStretchLastSection(True)
        header = fit_error.verticalHeader()
        header.setVisible(True)
        header.setDefaultSectionSize(25)
        header.setMinimumSectionSize(25)
        header.setStretchLastSection(True)

        i: int
        param: TAU_PARAMETER_NAME
        for i, param in enumerate(get_args(TAU_PARAMETER_NAME)):
            item: QTableWidgetItem = QTableWidgetItem()
            
            item.setText(str(param))
            fit_error.setHorizontalHeaderItem(i, item)
        fit_error.setHorizontalHeaderItem(i+1, QTableWidgetItem("Residual"))

        left_layout.addWidget(fit_error)
        self.fit_error = fit_error

        self._hidden_m = None
        self._m = None
        self._surface = None
        self._current = None
        self._saved = None
        right_layout: QVBoxLayout = QVBoxLayout()
        self.canvas_3d: MplCanvas3D = MplCanvas3D(self, width=5, height=4, dpi=100)
        right_layout.addWidget(self.canvas_3d)

        self.picker_radius = 25
        self._slice_m = None
        self._slice_hidden_m = None
        self._direct = None
        self._orbach = None
        self._raman = None
        self._raman_2 = None
        self._qtm = None
        self._v_d = None
        self._sum = None
        self._saved_sum = None
        self.canvas_slice: MplCanvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_slice.mpl_connect('pick_event', self.on_click)
        self._last_event_time:float = time()
        right_layout.addWidget(self.canvas_slice)

        left: QWidget() = QWidget()
        left.setLayout(left_layout)
        horizontal_spliter.addWidget(left)
        right: QWidget = QWidget()
        right.setLayout(right_layout)
        horizontal_spliter.addWidget(right)
        horizontal_spliter.setStretchFactor(0,1)
        horizontal_spliter.setStretchFactor(1,5)

        horizontal_layout.addWidget(horizontal_spliter)
        self.setLayout(horizontal_layout)

        self.varying_slice_combo_box.currentTextChanged.connect(self.set_model_varying)
        self.constant_value_slice.currentTextChanged.connect(self.set_model_constant)
        self.fit_button.clicked.connect(self.make_auto_fit)
        self.fit_for_slice_button.clicked.connect(self.make_auto_fit_for_slice)
        self.save_button.clicked.connect(self.save)
        self.reset_button.clicked.connect(self.reset)
        self.copy_parameters_button.clicked.connect(self.copy_parameters)
        self.adjust_range_button.clicked.connect(self.on_adjust_range)
        self.scale_checkbox.stateChanged.connect(self._update_fit_plot)

        self.shortcut: QShortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut.activated.connect(self.on_undo)

        self.shortcut_redo: QShortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.on_redo)

    def on_undo(self):
        if self.tau_fit is not None:
            self.tau_fit.undo()

    def on_redo(self):
        if self.tau_fit is not None:
            self.tau_fit.redo()

    def on_adjust_range(self):
        if self.tau_fit is not None:
            self.tau_fit.adjust_parameters()

    def make_auto_fit(self):
        self.tau_fit.make_auto_fit()

    def make_auto_fit_for_slice(self):
        if len(self.tau_fit.get_all_s()[0]) == 0:
            return

        self.tau_fit.make_auto_fit(slice_flag=True)

    def save(self):
        self.tau_fit.save()

    def reset(self):
        self.tau_fit.reset()

    def on_click(self, event):
        mouse = event.mouseevent

        tmp_time:float = time()
        if tmp_time - self._last_event_time < 0.1:
            return
        try:
            x_data = event.artist.get_xdata().values.tolist()
            y_data = event.artist.get_ydata().values.tolist()
        except AttributeError:
            x_data = event.artist.get_xdata()
            y_data = event.artist.get_ydata()

        ind = event.ind[0]
        if mouse.button == mouse.button.LEFT:
            self.tau_fit.hide_point(x_data[ind], y_data[ind])

        if mouse.button == mouse.button.RIGHT:
            self.tau_fit.delete_point(x_data[ind], y_data[ind])

        self._last_event_time = time()


    def on_constant_value_changed(self):
        self._update_plot_varying()

        self._update_measurements_plots()
        
        self._update_fit_plot()


    def set_model_varying(self, s: str):
        if s == "Field" or s == "Temperature":
            self.tau_fit.set_varying(cast(Literal['Field', 'Temperature'], s))

    def set_model_constant(self, s:str):
        if len(s.split()) == 2:
            self.tau_fit.set_constant(float(s.split()[0]))
        else:
            ...
            # print("Error")
            # print("S: ", s)

    def set_tau_fit(self, tau_fit:TauFit):
        if self.tau_fit is not None:
            for cid in self.cids:
                QObject.disconnect(cid)


        self.tau_fit = tau_fit
        

        i: int
        for i, p in enumerate(self.tau_fit.parameters):
            self.sliders[i].set_parameter(p)
            if p.is_blocked_on_0:
                self.sliders[i].set_disabled(True)
            else:
                self.sliders[i].set_disabled(False)

        self.cids: list[QMetaObject.Connection] = []

        self.cids.append(self.tau_fit.name_changed.connect(lambda new_name: self.title_label.setText(new_name)))
        self.cids.append(self.tau_fit.varying_changed.connect(self._update_slice_varying))
        self.cids.append(self.tau_fit.constant_changed.connect(self.on_constant_value_changed))
        self.cids.append(self.tau_fit.points_changed.connect(self.on_points_changed))
        self.cids.append(self.tau_fit.parameters_saved.connect(self.on_parameters_saved))
        self.cids.append(self.tau_fit.all_parameters_changed.connect(self._update_errors))
        
        for p in self.tau_fit.parameters:
            self.cids.append(p.value_changed.connect(self._update_fit_plot))
            self.cids.append(p.value_changed.connect(self._update_errors))

        self.title_label.setText(tau_fit.name)

        self._update_errors()
        self._update_saved_errors()

        self._update_fit_plot()
        self._update_measurements_plots()
        self._update_slice_varying()
        self.canvas_3d.draw()
        self.canvas_slice.draw()
        l = self.canvas_slice.ax.legend()
        l.set_draggable(True)

    def on_parameters_saved(self):
        self._update_plot_varying()
        self._update_fit_plot()
        self._update_saved_errors()

    def on_points_changed(self):

        self._update_plot_varying()
        self.canvas_3d.draw()
        self._update_measurements_plots()
        self._update_fit_plot()

        self.canvas_slice.draw()
        l = self.canvas_slice.ax.legend()
        l.set_draggable(True)

        self.canvas_3d.draw()

    def _update_slice_varying(self):
        j: int
        for j in range(self.varying_slice_combo_box.count()):
            if self.varying_slice_combo_box.itemText(j) == self.tau_fit.varying:
                self.varying_slice_combo_box.setCurrentIndex(j)

        self.constant_slice_label.setText("Field:" if self.tau_fit.varying == "Temperature" else "Temperature:")

        self.constant_value_slice.clear()
        _, temp, field = self.tau_fit.get_all()

        tmp: list[float] = temp if self.tau_fit.varying == "Field" else field
        tmp = list(set(tmp))
        tmp.sort()
        unit: str = "K" if self.tau_fit.varying == "Field" else "Oe"
        unique_strs: list[str] = [f"{v} {unit}" for v in tmp]
        self.constant_value_slice.addItems(unique_strs)

        self._update_measurements_plots()
        self._update_plot_varying()
        self._update_fit_slice()
        self.canvas_3d.draw()
        self.canvas_slice.draw()
        l = self.canvas_slice.ax.legend()
        l.set_draggable(True)

        #Terible temporary/pernament fix
        self.on_points_changed()

    def _update_plot_varying(self):
        tau, tmp, field = self.tau_fit.get_all()
        tau = log(tau)
        tmp = [1/t for t in tmp]

        c3: MplCanvas = self.canvas_3d
        if self._surface is not None:
            self._surface.remove()

        const: float = self.tau_fit.constant
        if self.tau_fit.varying == "Temperature":
            range = (min(tmp), max(tmp))
            if range[0] == range[1]:
                range = c3.axes.get_xlim()
            xx, zz = meshgrid(range, (min(tau), max(tau)))
            yy = ones(xx.shape) * const
        else:
            range = (min(field), max(field))
            if range[0] == range[1]:
                range = c3.axes.get_ylim()

            yy, zz = meshgrid(range, (min(tau), max(tau)))
            if const == 0:
                xx = ones(yy.shape) * 0
            else:
                xx = ones(yy.shape) * 1/const

        self._surface = c3.axes.plot_surface(xx, yy, zz, color="y", alpha=0.2, label='slice')

        c: MplCanvas = self.canvas_slice
        c.ax.set_xlabel(c.field_label if self.tau_fit.varying == "Field" else c.temp_label)

    def _update_fit_plot(self):

        tau, temp, field = self.tau_fit.get_all()

        temp_domain = linspace(min(temp), max(temp), self.resolution)
        field_domain = linspace(min(field), max(field), self.resolution)
        X, Y = meshgrid(temp_domain, field_domain)
        Z = -log(TauFit.model(X, Y, *self.tau_fit.get_parameters_values()))
        S = -log(TauFit.model(X, Y, *self.tau_fit.get_saved_parameters_values()))

        if self._current is not None:
            self._current.remove()

        if self._saved is not None:
            self._saved.remove()

        temp_domain = tuple(1/t for t in temp_domain)
        X, Y = meshgrid(temp_domain, field_domain)
        self._saved = self.canvas_3d.axes.plot_wireframe(X, Y, S, rstride=1, cstride=1, color='k', label='saved', alpha=0.5)
        self._current = self.canvas_3d.axes.plot_wireframe(X, Y, Z, rstride=1, cstride=1, label='current', linestyles='-', color=mcolors.TABLEAU_COLORS["tab:green"], alpha=0.4)
        self.canvas_3d.axes.set_zlim3d(min(Z.min(), S.min(), min(log(tau))), max(Z.max(),S.max(), max(log(tau))))
        self.canvas_3d.axes.set_ylim3d(Y.min(), Y.max())
        self.canvas_3d.axes.set_xlim3d(X.min(), X.max())
        self.canvas_3d.draw()
        self._update_fit_slice()

    def _update_fit_slice(self):
        tau, tmp, field = self.tau_fit.get_all_s()

        if len(tmp) < 2:
            if self._direct is not None:
                self._direct.remove()

            if self._orbach is not None:
                self._orbach.remove()

            if self._raman is not None:
                self._raman.remove()

            if self._raman_2 is not None:
                self._raman_2.remove()

            if self._qtm is not None:
                self._qtm.remove()

            if self._v_d is not None:
                self._v_d.remove()

            if self._sum is not None:
                self._sum.remove()

            if self._saved_sum is not None:
                self._saved_sum.remove()

            self._direct = None
            self._orbach = None
            self._raman = None
            self._raman_2 = None
            self._qtm = None
            self._v_d = None
            self._sum = None
            self._saved_sum = None

            
            if len(tau) == 1:
                tau = tau[0]
                self._update_measurements_plots()
                self.canvas_slice.ax.set_ybound(log(tau) - 0.05 , log(tau) + 0.05)
                
                
            l = self.canvas_slice.ax.legend()
            l.set_draggable(True)
            self.canvas_slice.draw()

            return

        if self.tau_fit.varying == "Field":
            xx = linspace(min(field), max(field), self.resolution)
        else:
            xx = linspace(min(tmp), max(tmp), self.resolution)
            xx = [1/x for x in xx]

        tmp = Series(linspace(min(tmp), max(tmp), self.resolution))
        field = Series(linspace(min(field), max(field), self.resolution))

        p: tuple[float, ...] = self.tau_fit.get_parameters_values()
        direct = -log(TauFit.direct(tmp, field, p[0], p[1]))
        orbach = -log(TauFit.Orbach(tmp, p[10], p[11]))
        raman = -log(TauFit.Raman(tmp, p[5], p[6]))
        raman_2 = -log(TauFit.Raman_2(tmp, field, p[7], p[8], p[9]))
        qtm = -log(TauFit.qtm(field, p[2], p[3], p[4]))
        v_d = -log(TauFit.V_d(tmp, p[12], p[13]))
        sum = -log(TauFit.model(tmp, field, *p))
        sum_saved = -log(TauFit.model(tmp, field, *self.tau_fit.get_saved_parameters_values()))

        if self._direct is None:
            self._direct = self.canvas_slice.ax.plot(xx, direct, "y--", label="Direct process")[0]
        else:
            self._direct.set_xdata(xx)
            self._direct.set_ydata(direct)

        if self._orbach is None:
            self._orbach = self.canvas_slice.ax.plot(xx, orbach, "m--", label="Orbach")[0]
        else:
            self._orbach.set_xdata(xx)
            self._orbach.set_ydata(orbach)

        if self._raman is None:
            self._raman = self.canvas_slice.ax.plot(xx, raman, "r--", label="Raman")[0]
        else:
            self._raman.set_xdata(xx)
            self._raman.set_ydata(raman)

        if self._raman_2 is None:
            self._raman_2 = self.canvas_slice.ax.plot(xx, raman_2, "c--", label="Raman_2")[0]
        else:
            self._raman_2.set_xdata(xx)
            self._raman_2.set_ydata(raman_2)

        if self._qtm is None:
            self._qtm = self.canvas_slice.ax.plot(xx, qtm, "g--", label="QTM")[0]
        else:
            self._qtm.set_xdata(xx)
            self._qtm.set_ydata(qtm)

        if self._v_d is None:
            self._v_d = self.canvas_slice.ax.plot(xx, v_d, "--", label="V_d", color=mcolors.TABLEAU_COLORS["tab:pink"])[0]
        else:
            self._v_d.set_xdata(xx)
            self._v_d.set_ydata(v_d)
        
        if self._sum is None:
            self._sum = self.canvas_slice.ax.plot(xx, sum, "b-", label="Current sum")[0]
        else:
            self._sum.set_xdata(xx)
            self._sum.set_ydata(sum)

        if self._saved_sum is None:
            self._saved_sum = self.canvas_slice.ax.plot(xx, sum_saved, "k-", label="Saved sum")[0]
        else:
            self._saved_sum.set_xdata(xx)
            self._saved_sum.set_ydata(sum_saved)

        min_y: float = min(log(tau))
        max_y: float = max(log(tau))
        if self.scale_checkbox.isChecked():
            all_values:list[float] = concatenate((direct, orbach, raman, raman_2, qtm, v_d, sum, sum_saved), axis=None)
            all_values = [v for v in all_values if not isnan(v) and not isinf(v)]
            if len(all_values) != 0:
                min_y = min(min(all_values), min_y)
                max_y = max(max(all_values), max_y)

        span_y: float = max_y - min_y
        if span_y == 0.0:
            span_y = 0.1
        self.canvas_slice.ax.set_ybound(min_y - span_y*0.05 , max_y + span_y*0.05)

        self.canvas_slice.draw()
        l = self.canvas_slice.ax.legend()
        l.set_draggable(True)
        return
        
    def _update_measurements_plots(self):
        if self.tau_fit is None:
            return

        if not(self._hidden_m is None or self._m is None):
            self._hidden_m.remove()
            self._m.remove()

        hidden_tau, hidden_temp, hidden_field = self.tau_fit.get_hidden()
        hidden_temp_invert = [1/t for t in hidden_temp]
        self._hidden_m = self.canvas_3d.axes.scatter(hidden_temp_invert, hidden_field, log(hidden_tau), "o",
            label='hidden points from fits', color=mcolors.TABLEAU_COLORS["tab:orange"])


        tau, temp, field = self.tau_fit.get_visible()
        temp_invert = [1/t for t in temp]
        self._m = self.canvas_3d.axes.scatter(temp_invert, field, log(tau), "o", color=mcolors.TABLEAU_COLORS["tab:blue"], label='points from fits')

        if self._slice_m is None or self._slice_hidden_m is None:
            xx: list[float]
            zz, temp, field = self.tau_fit.get_visible_s()
            if self.tau_fit.varying == "Field":
                xx = field
            else:
                xx = [1/t for t in temp]
            self._slice_m = self.canvas_slice.ax.plot(xx, log(zz), "o", c=mcolors.TABLEAU_COLORS["tab:blue"], picker=self.picker_radius)[0]

            zz, temp, field = self.tau_fit.get_hidden_s()
            if self.tau_fit.varying == "Field":
                xx = field
            else:
                xx = [1/t for t in temp]
            self._slice_hidden_m = self.canvas_slice.ax.plot(xx, log(zz), "o", c=mcolors.TABLEAU_COLORS["tab:orange"], picker=self.picker_radius)[0]
        else:
            zz, temp, field = self.tau_fit.get_visible_s()
            zz_h, temp_h, field_h = self.tau_fit.get_hidden_s()
            if self.tau_fit.varying == "Field":
                xx = field
                xx_h = field_h
            else:
                xx = [1/t for t in temp]
                xx_h = [1/t for t in temp_h]

            zz = log(zz)
            zz_h = log(zz_h)
            self._slice_m.set_xdata(xx)
            self._slice_m.set_ydata(zz)
            
            self._slice_hidden_m.set_xdata(xx_h)
            self._slice_hidden_m.set_ydata(zz_h)

            min_x:float = None
            max_x:float = None
            if len(xx) != 0:
                min_x = min(xx)
                max_x = max(xx)
            if len(xx_h) != 0:
                if min_x is not None:
                    min_x = min(min_x, min(xx_h))
                    max_x = max(max_x, max(xx_h))
            if min_x is not None and max_x is not None:
                span_x: float = max_x - min_x
                self.canvas_slice.ax.set_xbound(min_x - span_x*0.05 , max_x + span_x*0.05)

            min_z:float = None
            max_z:float = None
            if len(zz) != 0:
                min_z = min(zz)
                max_z = max(zz)
            if len(zz_h) != 0:
                if min_z is not None:
                    min_z = min(min_z, min(zz_h))
                    max_z = max(max_z, max(zz_h))
            if min_z is not None and max_z is not None:
                span_z: float = max_z - min_z
                self.canvas_slice.ax.set_ybound(min_z - span_z*0.05, max_z + span_z*0.05)


    def _update_errors(self):
        i:int
        for i, p in enumerate(self.tau_fit.parameters):
            self.fit_error.setItem(1, i, QTableWidgetItem(f"{p.value:.{8}e}  \u00b1 {p.error:.{8}e} "))
        self.fit_error.setItem(1, i+1, QTableWidgetItem(f"{self.tau_fit.residual_error:.{8}e}"))
    
    def _update_saved_errors(self):
        i:int
        for i, p in enumerate(self.tau_fit.saved_parameters):
            self.fit_error.setItem(0, i, QTableWidgetItem(f"{p.value:.{8}e} \u00b1 {p.error:.{8}e}"))
        self.fit_error.setItem(0, i+1, QTableWidgetItem(f"{self.tau_fit.saved_residual_error:.{8}e}"))

    def copy_parameters(self):
        dlg: QDialog = QDialog()
        dlg.setWindowTitle("Choose tau fit item to copy from")
        layout = QHBoxLayout()
        button = QPushButton("Copy")
        combo_box: QComboBox = QComboBox()
        layout.addWidget(combo_box)
        layout.addWidget(button)
        dlg.setLayout(layout)

        for n in self.tau_fit._collection.get_names():
            combo_box.addItem(n)

        button.clicked.connect(lambda: self.copy(combo_box.currentText(), dlg))
        dlg.exec()

    def copy(self, src_name: str, dlg: QDialog):
        other = self.tau_fit._collection.get_item_model(src_name)
        if other is not None:
            self.tau_fit.copy(other)
        dlg.close()
