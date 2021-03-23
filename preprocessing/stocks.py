import os
import pickle
import numpy as np
import scipy.io as sio

from preprocessing.data import Data


class StockData(Data):

    datafiles_path = os.path.join(os.path.dirname(__file__), "data_files")

    @staticmethod
    def get_data():
        filepath = os.path.join(StockData.datafiles_path, "snp452-data.mat")
        loaded_data = sio.loadmat(filepath)
        return loaded_data

    @staticmethod
    def get_stock_details(data):
        stock_info = data["stock"]
        stock_details = {}
        for i, x in enumerate(stock_info[0]):
            stock_details[i] = {
                "company_code": x[0, 0][0][0],
                "company_name": x[0, 0][1][0],
                "company_sector": x[0, 0][2][0]
            }
        return stock_details

    @staticmethod
    def get_stock_prices(data):
        return data["X"]

    @staticmethod
    def normalize_stock_price_diff(stock_prices):
        price_diff = np.diff(stock_prices, axis=0)
        mean_price_diff = np.mean(price_diff, axis=0)
        std_price_diff = np.std(price_diff, axis=0)
        normalized_price_diff = (price_diff - mean_price_diff) / std_price_diff
        return normalized_price_diff

    @staticmethod
    def save_data(data, filename, format="npy"):
        if format == "pkl":
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        elif format == "npy":
            np.save(filename, data)

    @staticmethod
    def load_data():
        datafiles = os.listdir(StockData.datafiles_path)
        if "stock_prices" not in datafiles or "stock_details" not in datafiles:
            data = StockData.get_data()

            stock_prices = StockData.get_stock_prices(data)
            filename = os.path.join(StockData.datafiles_path, "stock_prices")
            StockData.save_data(stock_prices, filename)

            stock_details = StockData.get_stock_details(data)
            filename = os.path.join(StockData.datafiles_path, "stock_details.pkl")
            StockData.save_data(stock_details, filename, format="pkl")

        else:
            with open(os.path.join(StockData.datafiles_path, "stock_details.pkl", "rb")) as f:
                stock_details = pickle.load(f)
            stock_prices = np.load(os.path.join(StockData.datafiles_path, "stock_prices.npy"))

        return stock_prices, stock_details
