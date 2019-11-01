from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, wavelet_features
from astronomaly.dimensionality_reduction import decomposition
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_dir = '/home/michelle/BigData/Anomaly/astronomaly_output/'


image_dataset = image_reader.ImageDataset('/home/michelle/BigData/Anomaly/Meerkat_deep2/',
                                          transform_function=image_preprocessing.image_transform_log,
                                          window_size=128, output_dir=output_dir)


# features = test_func(image_dataset).result()
# pipeline_psd = power_spectrum.PSD_Features(force_rerun=True, output_dir=output_dir)
# features = pipeline_psd.run_on_dataset(image_dataset)

pipeline_wavelets = wavelet_features.WaveletFeatures(force_rerun=True, output_dir=output_dir)
# new_func = python_app()(pipeline_wavelets.run_on_dataset)
features = pipeline_wavelets.run_on_dataset(image_dataset)
# features = new_func(image_dataset)

# print(image_dataset.get_display_image('0'))
#
# np.random.seed(0)
# dat = np.random.randn(1000,2)
# dat_outliers = np.random.randn(10,2)*5
#
# df = pd.DataFrame(data=np.vstack((dat, dat_outliers)), index=np.arange(1010).astype('str'))
# # plt.figure()
# # plt.plot(dat[:,0], dat[:,1],'.')
# # plt.plot(dat_outliers[:,0], dat_outliers[:,1],'.')
# # plt.show()
#
pipeline_pca = decomposition.PCA_Decomposer(n_components=2, output_dir=output_dir)
output_pca = pipeline_pca.run(features)
print('PCA output')
print(output_pca)
print()

pipeline_scaler = scaling.FeatureScaler(output_dir=output_dir)
output_scaled = pipeline_scaler.run(output_pca)
print('Scaled features')
print(output_scaled)
print('mean',output_scaled.mean().values)
print('std dev',output_scaled.std().values)
print()

pipeline_iforest = isolation_forest.IforestAlgorithm(output_dir=output_dir)
anomalies = pipeline_iforest.run(output_scaled)
print('10 most anomalous')
print(anomalies.sort_values('score')[:10])

pipeline_score_converter = human_loop_learning.ScoreConverter(output_dir=output_dir)
anomalies = pipeline_score_converter.run(anomalies)
anomalies = anomalies.sort_values('score', ascending=False)
print('10 most anomalous scaled')
print(anomalies[:10])

plt.figure()
plt.imshow(image_dataset.get_sample(anomalies.index[0]))
plt.show()

# pipeline_svd = decomposition.Truncated_SVD_Decomposer(n_components=3)
# out = pipeline_svd.run(df)
# print('Truncated SVD output')
# print(out)