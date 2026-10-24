"""
Minimal PySide6 UI for the ViralNet model.

Usage:
    python ui.py
"""
import sys

import pandas as pd
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from data import NUMERIC, BINARY, CATEGORICAL, _transform
from model import ViralNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints\\model.pt"


class ViralUI(QMainWindow):
    def __init__(self, ckpt_path: str = CKPT_PATH):
        super().__init__()
        self.setWindowTitle("Viral Video Predictor")
        self.setMinimumWidth(520)

        # ---- load model ----
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.model = ViralNet(ckpt["in_dim"]).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.encoders = ckpt["encoders"]
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.categories = self.encoders["cats"]["category"]
        self.days = self.encoders["cats"]["upload_day"]

        # ---- build UI ----
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # header
        title = QLabel("Viral Video Predictor")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel(f"model: {ckpt_path}  |  threshold: {self.threshold:.3f}  |  device: {DEVICE}")
        subtitle.setStyleSheet("color: #666; font-size: 11px;")
        subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(subtitle)

        # form
        form_box = QGroupBox("Input features")
        form = QFormLayout(form_box)

        self.in_category = QComboBox()
        self.in_category.addItems(self.categories)

        self.in_day = QComboBox()
        self.in_day.addItems(self.days)

        self.in_duration = QSpinBox()
        self.in_duration.setRange(0, 100_000)
        self.in_duration.setValue(300)
        self.in_duration.setSuffix(" sec")

        self.in_title_len = QSpinBox()
        self.in_title_len.setRange(0, 500)
        self.in_title_len.setValue(50)

        self.in_face = QComboBox()
        self.in_face.addItems(["No face (0)", "Face (1)"])

        self.in_hour = QSpinBox()
        self.in_hour.setRange(0, 23)
        self.in_hour.setValue(18)
        self.in_hour.setSuffix(" h")

        self.in_tags = QSpinBox()
        self.in_tags.setRange(0, 1000)
        self.in_tags.setValue(15)

        self.in_desc_len = QSpinBox()
        self.in_desc_len.setRange(0, 100_000)
        self.in_desc_len.setValue(300)

        self.in_subs = QSpinBox()
        self.in_subs.setRange(0, 2_000_000_000)
        self.in_subs.setSingleStep(10_000)
        self.in_subs.setValue(500_000)
        self.in_subs.setGroupSeparatorShown(True)

        form.addRow("Category:", self.in_category)
        form.addRow("Upload day:", self.in_day)
        form.addRow("Duration:", self.in_duration)
        form.addRow("Title length:", self.in_title_len)
        form.addRow("Thumbnail face:", self.in_face)
        form.addRow("Upload hour:", self.in_hour)
        form.addRow("Tags count:", self.in_tags)
        form.addRow("Description length:", self.in_desc_len)
        form.addRow("Subscriber count:", self.in_subs)

        root.addWidget(form_box)

        # predict button
        self.btn = QPushButton("Predict")
        self.btn.setMinimumHeight(38)
        self.btn.clicked.connect(self.on_predict)
        root.addWidget(self.btn)

        # result
        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; padding: 12px;"
            "border-radius: 8px; background: #f0f0f0; color: #333;"
        )
        root.addWidget(self.result_label)

        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet("color: #666; font-size: 12px;")
        root.addWidget(self.detail_label)

    def on_predict(self):
        row = {
            "category": self.in_category.currentText(),
            "upload_day": self.in_day.currentText(),
            "duration_sec": self.in_duration.value(),
            "title_length": self.in_title_len.value(),
            "thumbnail_face": 1 if self.in_face.currentIndex() == 1 else 0,
            "upload_hour": self.in_hour.value(),
            "tags_count": self.in_tags.value(),
            "description_length": self.in_desc_len.value(),
            "subscriber_count": self.in_subs.value(),
        }
        df = pd.DataFrame([row])

        try:
            X = _transform(df, self.encoders).to(DEVICE)
            with torch.no_grad():
                proba = torch.sigmoid(self.model(X)).item()
        except Exception as e:
            self.result_label.setText("Error")
            self.detail_label.setText(str(e))
            return

        is_viral = proba >= self.threshold
        if is_viral:
            self.result_label.setText(f"🔥 VIRAL  —  {proba:.1%}")
            self.result_label.setStyleSheet(
                "font-size: 22px; font-weight: bold; padding: 12px;"
                "border-radius: 8px; background: #ffe0e0; color: #b00020;"
            )
        else:
            self.result_label.setText(f"Not viral  —  {proba:.1%}")
            self.result_label.setStyleSheet(
                "font-size: 22px; font-weight: bold; padding: 12px;"
                "border-radius: 8px; background: #e6f4ea; color: #1b5e20;"
            )

        self.detail_label.setText(
            f"P(viral) = {proba:.3f}    threshold = {self.threshold:.3f}"
        )


def main():
    app = QApplication(sys.argv)
    win = ViralUI(CKPT_PATH)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()