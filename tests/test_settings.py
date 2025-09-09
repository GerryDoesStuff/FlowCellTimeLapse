import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings
from app.ui.main_window import MainWindow


def test_settings_persist(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    s = QSettings("YeastLab", "FlowcellPyQt")
    s.clear()
    s.sync()

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.max_iters.setValue(321)
    win.seg_method.setCurrentText("manual")
    win.ref_combo.setCurrentText("first")
    win.close()
    app.processEvents()

    win2 = MainWindow()
    assert win2.max_iters.value() == 321
    assert win2.seg_method.currentText() == "manual"
    assert win2.ref_combo.currentText() == "first"
    win2.close()
    app.quit()
