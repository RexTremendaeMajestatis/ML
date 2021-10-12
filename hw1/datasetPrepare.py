import scaling
import loader
import pandas as pd

def purgeDataset(dataset: pd.DataFrame):
    for i in dataset.columns:
        if (dataset[i] == 0).all():
            dataset.drop(i, inplace=True, axis=1)

# Remove zero columns from dataset
dataset5 = loader.loadTrainingDataset('dataset1.csv').append(loader.loadTrainingDataset('dataset2.csv')).append(loader.loadTrainingDataset('dataset3.csv')).append(loader.loadTrainingDataset('dataset4.csv'))
purgeDataset(dataset5)
test_dataset5 = loader.loadTrainingDataset('dataset5.csv')
purgeDataset(test_dataset5)

dataset4 = loader.loadTrainingDataset('dataset1.csv').append(loader.loadTrainingDataset('dataset2.csv')).append(loader.loadTrainingDataset('dataset3.csv')).append(loader.loadTrainingDataset('dataset5.csv'))
purgeDataset(dataset4)
test_dataset4 = loader.loadTrainingDataset('dataset4.csv')
purgeDataset(test_dataset4)

dataset3 = loader.loadTrainingDataset('dataset1.csv').append(loader.loadTrainingDataset('dataset2.csv')).append(loader.loadTrainingDataset('dataset4.csv')).append(loader.loadTrainingDataset('dataset5.csv'))
purgeDataset(dataset3)
test_dataset3 = loader.loadTrainingDataset('dataset3.csv')
purgeDataset(test_dataset3)

dataset2 = loader.loadTrainingDataset('dataset1.csv').append(loader.loadTrainingDataset('dataset3.csv')).append(loader.loadTrainingDataset('dataset4.csv')).append(loader.loadTrainingDataset('dataset5.csv'))
purgeDataset(dataset2)
test_dataset2 = loader.loadTrainingDataset('dataset2.csv')
purgeDataset(test_dataset2)

dataset1 = loader.loadTrainingDataset('dataset3.csv').append(loader.loadTrainingDataset('dataset2.csv')).append(loader.loadTrainingDataset('dataset4.csv')).append(loader.loadTrainingDataset('dataset5.csv'))
purgeDataset(dataset1)
test_dataset1 = loader.loadTrainingDataset('dataset1.csv')
purgeDataset(test_dataset1)

# Normalize dataset
dataset1 = scaling.normalize(dataset1)
test_dataset1 = scaling.normalize(test_dataset1)

dataset2 = scaling.normalize(dataset2)
test_dataset2 = scaling.normalize(test_dataset2)

dataset3 = scaling.normalize(dataset3)
test_dataset3 = scaling.normalize(test_dataset3)

dataset4 = scaling.normalize(dataset4)
test_dataset4 = scaling.normalize(test_dataset4)

dataset5 = scaling.normalize(dataset5)
test_dataset5 = scaling.normalize(test_dataset5)

# Save dataset to file
loader.saveNormalizedDataset(dataset1, 'dataset1.csv')
loader.saveNormalizedDataset(test_dataset1, 'test_dataset1.csv')

loader.saveNormalizedDataset(dataset2, 'dataset2.csv')
loader.saveNormalizedDataset(test_dataset2, 'test_dataset2.csv')

loader.saveNormalizedDataset(dataset3, 'dataset3.csv')
loader.saveNormalizedDataset(test_dataset3, 'test_dataset3.csv')

loader.saveNormalizedDataset(dataset4, 'dataset4.csv')
loader.saveNormalizedDataset(test_dataset4, 'test_dataset4.csv')

loader.saveNormalizedDataset(dataset5, 'dataset5.csv')
loader.saveNormalizedDataset(test_dataset5, 'test_dataset5.csv')