from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_files('hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images', path='C:\\Users\\sreec\\Desktop', unzip=True)
