import pandas as pd
import numpy as np
from typing import Mapping
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone
import shap

def spear_corr(df, y):
    spearman_corr = dict()
    X_cols = [col for col in df.columns if col != y]
    target_rank = df[y].rank(axis=0)
    for col in X_cols:
        col_rank = df[col].rank(axis=0)
        cov = col_rank.cov(target_rank)
        std_target, std_col = target_rank.std(), col_rank.std()
        spearman_corr[col] = abs(cov / (std_target * std_col))
    spearman_corr = {k: v for k, v in sorted(spearman_corr.items(), key=lambda item: abs(item[1]),reverse=True)}
    return spearman_corr


def pca_feature_imp(importances1, importances2, title1, title2):
    # draw a horizontal bar plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,12))
    ax[0].barh(np.array([val for val in importances1.keys()]),
               np.array([val for val in importances1.values()]), color='#30c9b5')
    ax[1].barh(np.array([val for val in importances2.keys()]),
               np.array([val for val in importances2.values()]), color='#30c9b5')

    # Remove x,y Ticks
    ax[0].xaxis.set_ticks_position('none')
    ax[0].yaxis.set_ticks_position('none')
    ax[1].xaxis.set_ticks_position('none')
    ax[1].yaxis.set_ticks_position('none')

    # Remove axes splines
    for s in ['top','bottom','left','right']:
        ax[0].spines[s].set_visible(False)
        ax[1].spines[s].set_visible(False)

    # Add x,y gridlines
    ax[0].grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    ax[1].grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    # Set the biggest value on the top
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    # Add Plot Title
    ax[0].set_title(title1, loc='left', pad=-5, fontweight='bold', fontsize=15, color = 'grey')
    ax[1].set_title(title2, loc='left', pad=-5, fontweight='bold', fontsize=15, color = 'grey')
    plt.tight_layout()
    return plt


def dropcol_importances(model,X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    baseline = model.oob_score_
    imp = {}
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = clone(model)
        model_.random_state = 2
        model_.fit(X_train_, y_train)
        m = model_.oob_score_
        imp[col] = baseline - m
    drop_imp = {k: v for k, v in sorted(imp.items(), key=lambda item: item[1],reverse=True)}
    return drop_imp

def permutation_importances(model, X_valid, y_valid, metric=accuracy_score):
    baseline = metric(y_valid, model.predict(X_valid))
    imp = {}
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        m = metric(y_valid, model.predict(X_valid))
        X_valid[col] = save
        imp[col] = baseline - m
    perm_imp = {k: v for k, v in sorted(imp.items(), key=lambda item: item[1],reverse=True)}
    return perm_imp


def compare_13(model, train, val, train_y, val_y, feat_imp, metric=log_loss):
    losses = []
    for i in range(1, 14):
        model_ = clone(model)
        model_.random_state = 2
        features = [col for col in feat_imp.keys()][:i]
        model_.fit(train.loc[:, features], train_y)
        valid_prob = model_.predict_proba(val.loc[:, features])
        log_loss_val = metric(val_y, valid_prob)
        losses.append(log_loss_val)
    return losses


def dropcol_importances_XGB(model,X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    model.random_state = 2
    valid_prob = model.predict_proba(X_valid)
    baseline = log_loss(y_valid, valid_prob)  # Calculate log loss
    imp = {}
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = clone(model)
        model_.random_state = 2
        model_.fit(X_train_, y_train)
        valid_prob_ = model_.predict_proba(X_valid_)
        m = log_loss(y_valid, valid_prob_)
        imp[col] = baseline - m
    drop_imp = {k: v for k, v in sorted(imp.items(), key=lambda item: item[1],reverse=True)}
    return drop_imp


def get_best_model(model, train_x, val_x, train_y, valid_y, y, metric=log_loss):
    model_ = clone(model)
    model_.random_state = 2
    model_.fit(train_x, train_y)
    preds = model_.predict_proba(val_x)
    val_loss = metric(valid_y, preds)
    shap_imp = get_shap_imp(model_, train_x, val_x, valid_y)
    feat = [col for col in val_x.columns]
    removed_feat = []

    for i in range(len(val_x.columns)):
        remove_feat = feat[np.argmin(shap_imp)]
        removed_feat.append(remove_feat)
        selected_feat = [col for col in feat if col not in removed_feat]
        print(f'Round {i}: drop feature of: {remove_feat}')
        model_ = clone(model)
        model_.random_state = 2
        X_train_new = train_x.drop(columns=removed_feat)
        X_val_new = val_x.drop(columns=removed_feat)
        model_.fit(X_train_new, train_y)
        preds_new = model_.predict_proba(X_val_new)
        new_val_loss = metric(valid_y, preds_new)

        if new_val_loss > val_loss:
            removed_feat = [col for col in removed_feat if col != remove_feat]
            print(f'Loss before dropping: {val_loss:.3f}, loss after dropping: {new_val_loss:.3f}')
            print(f'Stopping iterations, because dropping increasing loss. \n\nFinally, we dropped features:{removed_feat}')
            return model_, removed_feat

        print(f'Loss before dropping: {val_loss:.3f}, loss after dropping: {new_val_loss:.3f}')
        shap_imp = get_shap_imp(model_, X_train_new, X_val_new, valid_y)
        print(f'Loss decreased. Let us continue!','\n')
        feat = [col for col in X_train_new.columns]
        val_loss = new_val_loss

def get_shap_imp(model_, train_x, val_x, valid_y):
    shap_values = shap.TreeExplainer(model_, data=train_x).shap_values(X=val_x, y=valid_y, check_additivity=False)
    shap_imp = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    return shap_imp


def get_std(model, train_x, val_x, valid_y, metric=log_loss):
    shap_imp_ = np.zeros((50,val_x.shape[1]))
    for i in range(50):
        idx = np.random.choice(range(val_x.shape[0]), size=int(val_x.shape[0]*2/3), replace=False)
        val_new = val_x.iloc[idx, :]
        shap_imp_[i] = get_shap_imp(model, train_x, val_new, valid_y)
    return np.std(shap_imp_, axis=0)


def p_values(model, train_x, train_y, val_x, valid_y, target, metric=log_loss):
    shap_imp = np.zeros((100,val_x.shape[1]))
    shap_baseline = get_shap_imp(model, train_x, val_x, valid_y)
    shap_baseline = shap_baseline / np.sum(shap_baseline)
    for i in range(100):
        Y_train = np.random.permutation(train_y)
        model_ = clone(model)
        model_.random_state = 2
        model_.fit(train_x, Y_train)
        shap_imp[i] = get_shap_imp(model_, train_x, val_x, valid_y)
        shap_imp[i] = shap_imp[i] / np.sum(shap_imp[i])
    diff = shap_imp - shap_baseline
    p_values = np.sum(diff >= 0, axis=0) / 100
    return p_values, shap_baseline, shap_imp
