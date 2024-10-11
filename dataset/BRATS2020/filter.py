import pandas


class BraTS2020Filter:
    def __init__(self, xlsx_path):
        self.df = pandas.read_excel(xlsx_path, sheet_name='Sheet1')

    def __call__(self, number) -> bool:
        arr = self.df.values[number]
        return arr[1] >= 2 and arr[2] >= 2 and arr[3] >= 2
