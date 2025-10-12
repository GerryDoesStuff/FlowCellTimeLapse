from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolButton, QLayout


class CollapsibleSection(QWidget):
    """A simple collapsible container with a toggle header."""

    def __init__(
        self, title: str, parent: QWidget | None = None, collapsed: bool = False
    ) -> None:
        super().__init__(parent)
        self._button = QToolButton(self)
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(not collapsed)
        self._button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._button.setArrowType(
            Qt.ArrowType.DownArrow
            if not collapsed
            else Qt.ArrowType.RightArrow
        )
        self._button.clicked.connect(self._on_toggled)

        self._content = QWidget(self)
        self._content.setVisible(not collapsed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._button)
        layout.addWidget(self._content)

    def setContentLayout(self, layout: QLayout) -> None:
        """Set the layout that holds the section's child widgets."""
        self._content.setLayout(layout)

    def content(self) -> QWidget:
        """Return the inner content widget."""
        return self._content

    def _on_toggled(self) -> None:
        visible = self._button.isChecked()
        self._content.setVisible(visible)
        self._button.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self._content.updateGeometry()
