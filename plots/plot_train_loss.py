from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def dataframes(path):
    df = pd.read_csv(path, sep='\t', header=0)

    for col in df.columns[2:]:
        df[col] = [eval(x) for x in df[col]]

    df[['Loss - PA', 'Loss - NA', 'Loss - PI', 'Loss - NI']] = pd.DataFrame(df['Loss'].tolist(), index=df['Loss'].index)
    df[['Sensitivity - PA', 'Sensitivity - NA', 'Sensitivity - PI', 'Sensitivity - NI']] = pd.DataFrame(df['Sensitivity'].tolist(), index=df['Sensitivity'].index)
    df[['Specificity - PA', 'Specificity - NA', 'Specificity - PI', 'Specificity - NI']] = pd.DataFrame(df['Specificity'].tolist(), index=df['Specificity'].index)
    
    df['AUC - PA'] = pd.DataFrame(df['AUC - promoter active'].tolist(), index=df['AUC - promoter active'].index)[0]
    df['AUC - NA'] = pd.DataFrame(df['AUC - nonpromoter active'].tolist(), index=df['AUC - nonpromoter active'].index)[1]
    df['AUC - PI'] = pd.DataFrame(df['AUC - promoter inactive'].tolist(), index=df['AUC - promoter inactive'].index)[2]
    df['AUC - NI'] = pd.DataFrame(df['AUC - nonpromoter inactive'].tolist(), index=df['AUC - nonpromoter inactive'].index)[3]

    return df[df['Stage'] == 'train'], df[df['Stage'] == 'valid'] 


def prediction_dataframes(path):
    # Returns only the first row (epoch 0 results) of each dataframe
    df_train, df_valid = dataframes(path)
    return df_train[df_train.index == 0], df_valid[df_valid.index == 1]


def loss_dataframes(path):
    # Data frame for regression results: only one column – loss
    df = pd.read_csv(path, sep='\t', header=0)
    return df[df['Stage'] == 'train'], df[df['Stage'] == 'valid']  


def adjust_axes(ax, y_bottom, y_top, measure):
    # Adjust plot parameters
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_facecolor('gainsboro')
    ax.grid(color='white', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(y_bottom, y_top)
    ax.set_ylabel(f'{measure}')


def plot_predictions(data, title):
    train_dfs = []
    valid_dfs = []
    for path in data:
        df_train, df_valid = prediction_dataframes(path)
        train_dfs.append(df_train)
        valid_dfs.append(df_valid)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))

    measures = [('Loss', 0.97, 1.63), ('Sensitivity', -0.05, 1.05), ('Specificity', -0.05, 1.05), ('AUC', -0.05, 1.05)]
    x = np.arange(len(train_dfs))
    
    for i in range(len(measures)):
        measure, y_bottom, y_top = measures[i]
        y_pa_train, y_na_train, y_pi_train, y_ni_train = [], [], [], []
        y_pa_valid, y_na_valid, y_pi_valid, y_ni_valid = [], [], [], []

        for j in range(len(train_dfs)):
            y_pa_train.append(train_dfs[j][f'{measure} - PA']) 
            y_na_train.append(train_dfs[j][f'{measure} - NA']) 
            y_pi_train.append(train_dfs[j][f'{measure} - PI']) 
            y_ni_train.append(train_dfs[j][f'{measure} - NI']) 

            y_pa_valid.append(valid_dfs[j][f'{measure} - PA']) 
            y_na_valid.append(valid_dfs[j][f'{measure} - NA']) 
            y_pi_valid.append(valid_dfs[j][f'{measure} - PI']) 
            y_ni_valid.append(valid_dfs[j][f'{measure} - NI']) 

        pa, = axes[(0,i)].plot(x, y_pa_train, color='blue', marker='.', alpha=0.6)
        na, = axes[(0,i)].plot(x, y_na_train, color='orange', marker='.', alpha=0.6)
        pi, = axes[(0,i)].plot(x, y_pi_train, color='green', marker='.', alpha=0.6)
        ni, = axes[(0,i)].plot(x, y_ni_train, color='red', marker='.', alpha=0.6)

        pa, = axes[(1,i)].plot(x, y_pa_valid, color='blue', marker='.', alpha=0.6)
        na, = axes[(1,i)].plot(x, y_na_valid, color='orange', marker='.', alpha=0.6)
        pi, = axes[(1,i)].plot(x, y_pi_valid, color='green', marker='.', alpha=0.6)
        ni, = axes[(1,i)].plot(x, y_ni_valid, color='red', marker='.', alpha=0.6)
        

        for k in range(2):
            adjust_axes(axes[k,i], y_bottom, y_top, measure)
            axes[(k,i)].set_xlabel('Length in bp')
            axes[(k,i)].set_xticks(x)
            axes[(k,i)].set_xticklabels(['2000', '1600', '1200', '800', '600', '400', '200'], fontsize=8)
        
    axes[(0,2)].set_title('Training', x=-0.2, y=1.05)
    axes[(1,2)].set_title('Validation', x=-0.2, y=1.05)
    fig.suptitle(title, fontsize=16)

    plt.figlegend((pa, na, pi, ni), ('promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive'), loc='lower center', ncol=4, fontsize=8)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    plt.show()


def compare_predictions(data, title, subplots=None, last=False):    
    train_dfs = []
    valid_dfs = []
    for path in data:
        df_train, df_valid = prediction_dataframes(path)
        train_dfs.append(df_train)
        valid_dfs.append(df_valid)

    fig, axes = subplots if subplots else plt.subplots(2, 4, figsize=(13, 6))

    measures = [('Loss', 0.97, 1.63), ('Sensitivity', -0.05, 1.05), ('Specificity', -0.05, 1.05), ('AUC', -0.05, 1.05)]
    x = np.arange(len(train_dfs))
    
    for i in range(len(measures)):
        measure, y_bottom, y_top = measures[i]
        y_pa_train, y_na_train, y_pi_train, y_ni_train = [], [], [], []
        y_pa_valid, y_na_valid, y_pi_valid, y_ni_valid = [], [], [], []

        for j in range(len(train_dfs)):
            y_pa_train.append(train_dfs[j][f'{measure} - PA']) 
            y_na_train.append(train_dfs[j][f'{measure} - NA']) 
            y_pi_train.append(train_dfs[j][f'{measure} - PI']) 
            y_ni_train.append(train_dfs[j][f'{measure} - NI']) 

            y_pa_valid.append(valid_dfs[j][f'{measure} - PA']) 
            y_na_valid.append(valid_dfs[j][f'{measure} - NA']) 
            y_pi_valid.append(valid_dfs[j][f'{measure} - PI']) 
            y_ni_valid.append(valid_dfs[j][f'{measure} - NI']) 

        if last:
            pa, = axes[(0,i)].plot(x, y_pa_train, color='blue', alpha=0.9)
            na, = axes[(0,i)].plot(x, y_na_train, color='orange', alpha=0.9)
            pi, = axes[(0,i)].plot(x, y_pi_train, color='green', alpha=0.9)
            ni, = axes[(0,i)].plot(x, y_ni_train, color='red', alpha=0.9)

            pa, = axes[(1,i)].plot(x, y_pa_valid, color='blue', alpha=0.9)
            na, = axes[(1,i)].plot(x, y_na_valid, color='orange', alpha=0.9)
            pi, = axes[(1,i)].plot(x, y_pi_valid, color='green', alpha=0.9)
            ni, = axes[(1,i)].plot(x, y_ni_valid, color='red', alpha=0.9)  

        else:
  
            pa, = axes[(0,i)].plot(x, y_pa_train, color='blue', alpha=0.5)
            na, = axes[(0,i)].plot(x, y_na_train, color='orange', alpha=0.5)
            pi, = axes[(0,i)].plot(x, y_pi_train, color='green', alpha=0.5)
            ni, = axes[(0,i)].plot(x, y_ni_train, color='red', alpha=0.5)

            pa, = axes[(1,i)].plot(x, y_pa_valid, color='blue', alpha=0.5)
            na, = axes[(1,i)].plot(x, y_na_valid, color='orange', alpha=0.5)
            pi, = axes[(1,i)].plot(x, y_pi_valid, color='green', alpha=0.5)
            ni, = axes[(1,i)].plot(x, y_ni_valid, color='red', alpha=0.5) 

        for k in range(2):
            adjust_axes(axes[k,i], y_bottom, y_top, measure)
            axes[(k,i)].set_xlabel('Length in bp')
            axes[(k,i)].set_xticks(x)
            axes[(k,i)].set_xticklabels(['2000', '1600', '1200', '800', '600', '400', '200'], fontsize=8)
        
    axes[(0,2)].set_title('Training', x=-0.2, y=1.05)
    axes[(1,2)].set_title('Validation', x=-0.2, y=1.05)
    fig.suptitle(title, fontsize=16)

    plt.figlegend((pa, na, pi, ni), ('promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive'), loc='lower center', ncol=4, fontsize=8)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 

    if last:
        plt.show()

def plot_train_results(path, title, three_classes=False):
    df_train, df_valid = dataframes(path)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    # to do: (measure, ylim) list to adjust the y-axis
    measures = [('Loss', 0.97, 1.63), ('Sensitivity', -0.05, 1.05), ('Specificity', -0.05, 1.05), ('AUC', -0.05, 1.05)]
    dfs = df_train, df_valid

    for i in range(len(measures)):
        # Change dataframes to have important columns (instead of tuple columns)
        measure, y_bottom, y_top = measures[i]

        for j in range(len(dfs)):
            # Plot results for both datasets
            pa = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - PA'], color='blue', s=3, alpha=0.6)
            na = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - NA'], color='orange', s=3, alpha=0.6)
            pi = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - PI'], color='green', s=3, alpha=0.6)
            ni = axes[j,i].scatter(x = dfs[j]['Epoch'], y = dfs[j][f'{measure} - NI'], color='red', s=3, alpha=0.6)

            # Adjust plot parameters
            adjust_axes(axes[j,i], y_bottom, y_top, measure)
            axes[(j,i)].set_xlabel('Epoch')
        
    axes[(0,2)].set_title('Training', x=-0.2, y=1.05)
    axes[(1,2)].set_title('Validation', x=-0.2, y=1.05)
    fig.suptitle(title, fontsize=16)

    if three_classes:
        plt.figlegend((pa, na, pi, ni), ('active regions', 'silent regions', 'inactive regions', 'none'), loc='lower center', ncol=4, fontsize=8)
    else:
        plt.figlegend((pa, na, pi, ni), ('promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive'), loc='lower center', ncol=4, fontsize=8)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    plt.show()


def plot_loss(path, title):

    df_train, df_valid = loss_dataframes(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(x = df_train['Epoch'], y = df_train[f'Loss'], color='blue', s=3, alpha=0.6)
    axes[0].set_xlabel('Epoch')
    # axes[0].set_ylabel('MSE')
    # axes[0].set_title('Training', x=-0.2, y=1.05)
    axes[0].set_title('Training')
    # adjust_axes(axes[0], 0.7, 2.3, 'MSE')
    adjust_axes(axes[0], -0.2, 2.3, 'MSE')

    axes[1].scatter(x = df_valid['Epoch'], y = df_valid[f'Loss'], color='blue', s=3, alpha=0.6)
    axes[1].set_xlabel('Epoch')
    # axes[1].set_title('Validation', x=-0.2, y=1.05)
    axes[1].set_title('Validation')
    # adjust_axes(axes[1], 0.7, 2.3, 'MSE')
    adjust_axes(axes[1], -0.2, 2.3, 'MSE')
    fig.suptitle(title, fontsize=16)
    plt.show()


def plot_joint_loss(results1, results2, title):

    df_train1, df_valid1 = loss_dataframes(results1)
    df_train2, df_valid2 = loss_dataframes(results2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x = df_train1['Epoch'], y = df_train1[f'Loss'], s=3, alpha=0.6, label='training')
    axes[0].scatter(x = df_valid1['Epoch'], y = df_valid1[f'Loss'], s=3, alpha=0.6, label='validation')
    axes[0].set_xlabel('Epoch') 
    axes[0].set_title('Active sequences')
    adjust_axes(axes[0], -0.2, 2.3, 'MSE')

    train = axes[1].scatter(x = df_train2['Epoch'], y = df_train2[f'Loss'], s=3, alpha=0.6, label='training')
    val = axes[1].scatter(x = df_valid2['Epoch'], y = df_valid2[f'Loss'], s=3, alpha=0.6, label='validation')
    axes[1].set_xlabel('Epoch') 
    axes[1].set_title('Silent sequences')
    adjust_axes(axes[1], -0.2, 2.3, '')
    axes[1].legend()
    # plt.figlegend((train, val), ('training', 'validation'), ncol=1, fontsize=8)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) 
    fig.suptitle(title, fontsize=16)
    plt.show()



train_alt1_3_classes_ol = 'data/three_classes/ol_predictions/alt1_3_classes_ol_train_results.tsv'
# plot_train_results(train_alt1_3_classes_ol, 'Results of training Alt1 on different class division for original length sequences')

train_alt1_3_classes_min_200bp = 'data/three_classes/min_200bp_predictions/alt1_3_classes_min_200bp_train_results.tsv'
# plot_train_results(train_alt1_3_classes_min_200bp, 'Results of training Alt1 on different class division for shorter sequences elongated to 200bp')

if __name__ == '__main__':
    """
    regression_run1 = 'data/regression/train/regression-run1_train_results.tsv'
    plot_loss(regression_run1, 'Regression on sequences minimum 200 bp long with learning rate 0.001 (2022 dataset)')

    regression_lr_0_01 = 'data/regression/train/regression-run2_train_results.tsv'
    plot_loss(regression_lr_0_01, 'Regression on sequences minimum 200 bp long with learning rate 0.01 (2022 dataset)')

    regression_all_data = 'data/regression/train/regression-all_data_train_results.tsv'
    plot_loss(regression_all_data, 'Regression on all sequences from 2022')

    regression_no_model = 'data/regression/train/regression-no–model_train_results.tsv'
    plot_loss(regression_no_model, 'Regression on sequences minimum 200 bp long with no initial model provided (2022 dataset)')

    regression_all_data_no_model = 'data/regression/train/regression-all_data-no_model_train_results.tsv'
    plot_loss(regression_all_data_no_model, 'Regression on all sequences from 2022 with no initial model provided')

    regression_trial_2_2022 = 'data/regression/train_trial_2/regression-all_data_train_results.tsv'
    plot_loss(regression_trial_2_2022, '')
    """

    # PRÓBA 2

    reg_all_2022 = 'data/regression/train_real/regression-2022-all_data_train_results.tsv'
    plot_loss(reg_all_2022, 'Regression network training on 2022 data')

    reg_all_2022_no_model = 'data/regression/train_real/regression-2022-all_data-no_model_train_results.tsv'
    plot_loss(reg_all_2022_no_model, 'Regression network training on 2022 data with no initial model provided')

    reg_active_2022_no_model = 'data/regression/train_real/regression-2022-only_active-no_model_train_results.tsv'
    plot_loss(reg_active_2022_no_model, 'Regression network training on 2022 active sequences with no initial model')

    reg_silent_2022_no_model = 'data/regression/train_real/regression-2022-only_silent_train_results.tsv'
    plot_loss(reg_silent_2022_no_model, 'Regression network training on 2022 silent sequences with no initial model')

    reg_active_2022_alt1 = 'data/regression/train_real/regression-2022-only_active-alt1_train_results.tsv'
    plot_loss(reg_active_2022_alt1, 'Regression network training on 2022 active sequences')

    reg_active_2022_alt1_dropout = 'data/regression/train_trial2/regression-2022-active-alt1_dropout_train_results.tsv'
    plot_loss(reg_active_2022_alt1_dropout, 'Regression network training on 2022 active sequences')


    plot_joint_loss(reg_active_2022_no_model, reg_silent_2022_no_model, 'Regression models training on 2022 sequences')