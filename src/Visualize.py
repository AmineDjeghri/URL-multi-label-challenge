
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_target_distribution(df):
	"""
		Plot the global distribution ( or count ) of each target's in the entire dataset of lists.
	"""
	count_labels = df.explode('target')['target'].value_counts()
	df_stats = count_labels
	df_stats.index = pd.to_numeric(df_stats.index)
	df_stats.sort_index(inplace=True)

	sns.set(font_scale = 1)
	plt.figure(figsize=(15,8))

	plt.xlabel('targets')
	plt.ylabel('occurences')
	plt.title('Distribution des targets')

	df_stats.plot(figsize=(12,5) ) 

