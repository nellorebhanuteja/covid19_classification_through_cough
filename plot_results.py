import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
results_folder = 
systems = [
    'jio'
]
classifier = 'mlp' # lr, rf, mlp
features = [
    'mfcc'
]
# [
#     'fbank', 'fbank_delta', 'fbank_energy', 'fbank_mfcc', 'fbank_mfcc_energy', 'fbank_mfcc_energy_pitch', 'fbank_mfcc_vmd', 'fbank_vmd','mfcc', 'mfcc_energy', 'mfcc_energy_pitch', 'mfcc_vmd', 'vmd', 'vmd_delta', 'vmd_k3', 'vmd_k3_delta'
#     ]
def process_results(results, system):
    
    final_results = []
    for i in range(0,len(results)):
        temp=[]
        for j in range(1,len(results[i])):
            temp_new = numpy.array([results[i][0][:-1] , results[i][j], system])
            if j ==1:
                temp = temp_new
            else:
                temp = numpy.row_stack([temp, temp_new])
        if i==0:
            final_results = temp
        else:
            final_results = numpy.row_stack([final_results, temp])
    return final_results
for system in systems:
    if system == 'baseline':
        filename =  results_folder + 'results_' + system + '/' + 'results_' + classifier + '/val_summary_metrics.txt'
        results_baseline = numpy.genfromtxt(filename, dtype=None, encoding=None, skip_footer=3)
        results_baseline = process_results(results_baseline, 'Baseline')
    else:
        i=1 
        for feature in features:
            filename = results_folder + 'results_' + system + '/' + feature + '/' + 'results_' + classifier + '/val_summary_metrics.txt'
            results_jio_temp = numpy.genfromtxt(filename, dtype=None, encoding=None, skip_footer=3)
            results_jio_temp = process_results(results_jio_temp, feature)
            if i==1:
                results_jio = results_jio_temp
            else:
                results_jio = numpy.row_stack([results_jio, results_jio_temp])
            i+=1

#results = numpy.row_stack([results_baseline, results_jio])
results = results_jio
df = pandas.DataFrame(data=results, columns=["Measure", "Accuracy", "Feature Type"])
df = df.explode('Accuracy')
df['Accuracy'] = df['Accuracy'].astype('float')
title = {
    'lr': 'Linear Regression',
    'rf': 'Random Forest',
    'mlp': 'Multi Layer Perceptron',
    'svm': 'SVM'
}
sns.set_theme(style="whitegrid")
# g = sns.catplot(x="Measure", y="Accuracy", hue="Feature Type",data=df, saturation=.5,kind="bar", ci='sd', aspect=.6, legend_out=True)
g = sns.barplot(x = "Measure", y = "Accuracy", hue = "Feature Type", data = df)
g.set(ylim=(0, 100))
g.set(xlabel='', ylabel='Percentage (%)')
g.legend(loc='upper right')
g.set_title(title[classifier])
# (g.set_axis_labels("", "").set_titles("{col_name}").set(ylim=(0, 100))) 
plt.show()