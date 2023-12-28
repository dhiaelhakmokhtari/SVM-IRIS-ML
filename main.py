import sys
import joblib
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow
from interface import Ui_MainWindow  # Import from your generated code


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.calculate_button_clicked)

    def calculate_button_clicked(self):

        sepal_length = float(self.lineEdit.text())
        sepal_width = float(self.lineEdit_2.text())
        petal_length = float(self.lineEdit_3.text())
        petal_width = float(self.lineEdit_5.text())

        model_path = 'Model/iris classification best hyperparameters.pkl'
        svm_model = joblib.load(model_path)

        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
        result = svm_model.predict(input_data)
        self.label_6.setText(f"{int(result[0])}")
        if 0 in result:
            self.label_6.setText(f"Setosa")
        elif 1 in result:
            self.label_6.setText(f"Versicolor")
        elif 2 in result:
            self.label_6.setText(f"Virginica")
        else:
            print("Unknown label")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.setFixedSize(window.size())
    window.show()
    sys.exit(app.exec_())
