import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy as sp

def get_data():
    # read data and make new binary column
    data1 = pd.read_table('student-mat.csv', sep = ';')
    data2 = pd.read_table('student-por.csv', sep = ';')

    data = pd.concat([data1,data2])
    data = data[['sex','age','Pstatus', 'higher', 'internet', 'romantic', 'famrel', 'goout', 'Dalc', 'famsize',
    'absences']]

    data['absencebin'] = np.where(data['absences'] > 0, 'Absent Logged', 'Perfect Attendance')

    return data

def process_data(data):
    # process data, get predictors to (0,1)
    types = data.select_dtypes(include = ['object']).columns
    for col in types:
        data[col] = pd.get_dummies(data[col], drop_first = True)

## read in data

data = get_data()

## plotting

## histogram

ax = sns.displot(data, x= 'absences', bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,np.max(data['absences'])])
plt.title('Overall Distribution of Absences', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize =14)
plt.xlabel('Absences', fontsize=20)
plt.ylabel('Count', fontsize = 20)
plt.subplots_adjust(top = 0.9)

#boxplots

ax = sns.displot(x = 'famrel', hue = 'absencebin', multiple='stack', data = data)
ax.set(xlabel = 'Family Relationship (1-5)')
ax.fig.suptitle('Family Relationships, split by Perfect Attendance')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylabel('Count', fontsize = 20)
plt.subplots_adjust(top = 0.9)

ax = sns.displot(x = 'Pstatus', hue = 'absencebin', multiple='stack', data = data)
sns.set(font_scale = 1.5)

ax = sns.displot(data, x= 'absences', hue = 'romantic', multiple = 'stack', bins = [0,2,4,6,8,10,12,14,16,18,20,np.max(data['absences'])])
ax.fig.suptitle('Absences, Split by Romantic Relationships', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('Absences', fontsize=20)
plt.ylabel('Count', fontsize = 20)
plt.subplots_adjust(top = 0.9)

plt.figure(figsize = (8,6))

ax = sns.boxplot(data = data, x = 'romantic', y = 'absences')

## process the data for analysis now that we're done exploratory plotting

process_data(data)

## exploratory analysis

print('male mean absences: ' + str(data[data.sex == 1].absences.mean()))
print('female mean absences: ' + str(data[data.sex == 0].absences.mean()))

print('romantic: Y mean absences: ' + str(data[data.romantic == 1].absences.mean()))
print('romantic: N mean absences: ' + str(data[data.romantic == 0].absences.mean()))

print('parents cohabitate: Y mean absences: ' + str(data[data.Pstatus == 1].absences.mean()))
print('parents cohabitate: N mean absences: ' + str(data[data.Pstatus == 0].absences.mean()))

print('internet at home: Y mean absences: ' + str(data[data.internet == 1].absences.mean()))
print('internet at home: N mean absences: ' + str(data[data.internet == 0].absences.mean()))

print('less than 3 children: mean absences: ' + str(data[data.famsize == 1].absences.mean()))
print('more than 3 children: mean absences: ' + str(data[data.famsize == 0].absences.mean()))

print('male proportion of perfect attendance: ' + str(data[data.sex == 1].absencebin.mean()))
print('female proportion of perfect attendance: ' + str(data[data.sex == 0].absencebin.mean()))

print('romantic: Y proportion of perfect attendance: ' + str(data[data.romantic == 1].absencebin.mean()))
print('romantic: N proportion of perfect attendance: ' + str(data[data.romantic == 0].absencebin.mean()))

print('parents cohabitate: Y proportion of perfect attendance: ' + str(data[data.Pstatus == 1].absencebin.mean()))
print('parents cohabitate: N proportion of perfect attendance: ' + str(data[data.Pstatus == 0].absencebin.mean()))

print('internet at home: Y proportion of perfect attendance: ' + str(data[data.internet == 1].absencebin.mean()))
print('internet at home: N proportion of perfect attendance: ' + str(data[data.internet == 0].absencebin.mean()))

print('less than 3 children: Y proportion of perfect attendance: ' + str(data[data.famsize == 1].absencebin.mean()))
print('more than 3 children: N proportion of perfect attendance: ' + str(data[data.famsize == 0].absencebin.mean()))

print(data[data.absences>20].describe())
print(data.describe())

print(data[data.age>19].describe())

print('family relationship: 5 proportion of perf. attendance' + str(data[data.famrel == 5].absencebin.mean()))
print('family relationship: 4 proportion of perf. attendance' + str(data[data.famrel == 4].absencebin.mean()))
print('family relationship: 3 proportion of perf. attendance' + str(data[data.famrel == 3].absencebin.mean()))
print('family relationship: 2 proportion of perf. attendance' + str(data[data.famrel == 2].absencebin.mean()))
print('family relationship: 1 proportion of perf. attendance' + str(data[data.famrel == 1].absencebin.mean()))

print('family relationship: 5 avg attendance' + str(data[data.famrel == 5].absences.mean()))
print('family relationship: 4 avg attendance' + str(data[data.famrel == 4].absences.mean()))
print('family relationship: 3 avg attendance' + str(data[data.famrel == 3].absences.mean()))
print('family relationship: 2 avg attendance' + str(data[data.famrel == 2].absences.mean()))
print('family relationship: 1 avg attendance' + str(data[data.famrel == 1].absences.mean()))

## correlation plotting

sns.set(font_scale = 0.75)
resp = data['absences']
covariates = data.drop(['absencebin', 'absences'],axis = 1)
corr = np.abs(np.round(covariates.corr(),2))

sns.heatmap(corr,
            xticklabels = corr.columns,
            yticklabels = corr.columns,
            annot=True
            )

## separate data for regression

resp = data['absences']
covariates = data.drop(['absencebin', 'absences'],axis = 1)

covariates = sm.add_constant(covariates)

## define backward selection algorithm, adapted from activity

def aic(p, n):
    return len(p)
def bic(p, n):
    bicc = np.log(n)/2
    return bicc * (len(p))

def backward_selection(X, resp, penalty, fit_func):
    ## X covariate matrix
    ## resp response vector
    ## penalty penalty function(p, n)
    ## fit sm.Poisson(), or similar
    selected = list(X.columns)

    # compute objective for all columns
    model = fit_func(endog=resp, exog=X[selected])
    res = model.fit(maxiter=100, disp=False)
    curobj = res.llf - penalty(res.params, len(resp))
    while True:
        bestobj = -np.inf
        bestcoln = None
        # iterate over possible columns to remove
        for coln in selected:
            tmp_sel = selected.copy()
            tmp_sel.remove(coln)

            X_temp = X[tmp_sel]
            fit = (fit_func(resp, X_temp)).fit(disp=0)
            beta = fit.params
            tmpobj = fit.llf - penalty(beta, len(resp))

            if (tmpobj > bestobj) and (tmpobj > curobj):
                print(len(beta))
                bestcoln = coln
                bestobj = tmpobj

        # if removing at least one column improved the objective, remove it and loop; otherwise quit
        if bestcoln is not None:
            selected.remove(bestcoln)
            curobj = bestobj
            print(bestobj)
            print(selected)
        else:
            print('Done')
            break
    model = fit_func(endog=resp, exog=X[selected])
    res = model.fit(maxiter=100, disp=False)

    return curobj, selected, res

#fitting optimal poisson and negbin

obj_pois, selected_pois, fit_pois = backward_selection(X = covariates, resp = resp, penalty = bic, fit_func = sm.Poisson)
obj_negbin, selected_negbin, fit_negbin = backward_selection(X = covariates, resp = resp, penalty = bic, fit_func = sm.NegativeBinomial)

#computing standardized residuals
pred_pois = fit_pois.predict()
sd_pois_pred = np.sqrt(pred_pois)
scale_pois_res = (resp-pred_pois)/(sd_pois_pred)

pred_negbin = fit_negbin.predict()
sd_negbin_pred = (pred_negbin + np.power(pred_negbin, 2)*fit_negbin.params.alpha)
scale_negbin_res = (resp-pred_negbin)/(sd_negbin_pred)

#plotting residual plots

fig, (ax1, ax2) = plt.subplots(figsize = (8,6), ncols = 2)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)

plt.suptitle('Standardized Residuals', fontsize = 30)
ax1.scatter(pred_pois, scale_pois_res)
ax1.set_xlabel('Predicted Value', fontsize = 15)
ax1.set_ylabel('Standardized Residuals', fontsize = 20)
ax1.hlines(0, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1],
linestyles = '--', colors = 'r')
ax1.set_title('Poisson GLM', fontsize = 30)

ax2.scatter(pred_negbin, scale_negbin_res)
ax2.set_xlabel('Predicted Value', fontsize = 15)
ax2.hlines(0, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1],
linestyles = '--', colors = 'r')
ax2.set_title('NegBin GLM', fontsize = 30)

## computing probabilities of perfect attendance
pois_pred_0_0 = np.exp(-np.matmul(covariates[resp == 0][selected_pois], fit_pois.params))  ## mean for perfect attendance observations
print(pois_pred_0_0.mean())
pois_pred_0_all = np.exp(-np.matmul(covariates[selected_pois], fit_pois.params)) ## mean for all observations
print(pois_pred_0_all.mean())

nb_alpha = fit_negbin.params.alpha
negbin_mean = np.exp(np.matmul(covariates[selected_negbin], fit_negbin.params.drop('alpha')))
negbin_pred_0_0 = np.power(1 + nb_alpha*negbin_mean[resp==0], (-1/nb_alpha))
print(negbin_pred_0_0.mean())
negbin_pred_0_all = np.power(1 + nb_alpha*negbin_mean, (-1/nb_alpha))
print(negbin_pred_0_all.mean())

## print full fits

print(fit_pois.summary())
print(fit_negbin.summary())




### ZERO INFLATED MODELS

## define forward selection wtih zero inflation predictors

def forward_selection_ZIF(X, resp, penalty, fit_func):
    ## X covariate matrix
    ## resp response vector
    ## penalty penalty function(p, n)
    ## fit sm.ZeroInflatedPoisson(), or similar

    #define a list of predictors for each model
    selected_base = []
    selected_infl = []  ## logistic part
    unselected_base = list(X.drop('const', axis = 1).columns)
    unselected_infl = list(X.drop('const', axis = 1).columns)

    # make sure we add at least one column by setting curobj to -infinity
    curobj = -np.inf
    while True:
        bestobj = -np.inf
        bestcoln = None
        whichcoln = 'base'
        # iterate over possible columns to add
        for coln in unselected_base:
            tmp_sel_base = selected_base + [coln]
            if len(selected_infl) == 0:   ## if none selected yet from logistic,
                fit = (fit_func(endog=resp, exog=X[tmp_sel_base])).fit(disp=0)   ## default exog_infl fits with just the constant
            else:
                fit = (fit_func(endog=resp, exog=X[tmp_sel_base], exog_infl=X[selected_infl])).fit(disp=0)  # else use the current logistic selection
            beta = fit.params  ## this will have length of all predictors
            tmpobj = fit.llf - penalty(beta, len(resp))

            if (tmpobj > bestobj) and (tmpobj > curobj):
                bestcoln = coln
                bestobj = tmpobj
                whichcoln = 'base'  ## best column is from base model

        for coln in unselected_infl:
            tmp_sel_infl = selected_infl + [coln]
            if len(selected_base) == 0:  ## if none selected in base model yet,
                fit = (fit_func(endog=resp, exog=X[['const']], exog_infl = X[tmp_sel_infl])).fit(disp=0)  #pick best selected from above round
            else:
                fit = (fit_func(endog=resp, exog=X[selected_base], exog_infl=X[tmp_sel_infl])).fit(disp=0)
            beta = fit.params
            tmpobj = fit.llf - penalty(beta, len(resp))

            if (tmpobj > bestobj) and (tmpobj > curobj):
                bestcoln = coln
                bestobj = tmpobj
                whichcoln = 'infl'  #best column is from logistic model

        # if at least one column improved the objective, add it and loop; otherwise quit
        if bestcoln is not None:
            ## depending on where best column is from, add to the corresponding model
            if whichcoln == 'base':
                selected_base.append(bestcoln)
                unselected_base.remove(bestcoln)
            elif whichcoln == 'infl':
                selected_infl.append(bestcoln)
                unselected_infl.remove(bestcoln)

            curobj = bestobj
            print(bestobj)
            print(selected_base + selected_infl)
            print(whichcoln)
        else:
            print('Done')
            break
    model = fit_func(endog=resp, exog=X[selected_base], exog_infl = X[selected_infl])
    res = model.fit(maxiter=100, disp=False)
    return curobj, selected_base, selected_infl, res

def backward_selection_ZIF(X, resp, penalty, fit_func):
    ## X covariate matrix
    ## resp response vector
    ## penalty penalty function(p, n)
    ## fit sm.ZeroInflatedPoisson(), or similar

    #define separately for each
    selected_base = list(X.columns)
    selected_infl = list(X.columns)

    # compute objective for all columns
    model = fit_func(endog=resp, exog=X[selected_base], exog_infl = X[selected_infl])
    res = model.fit(maxiter=100, disp=False)
    curobj = res.llf - penalty((selected_base+selected_infl), len(resp))
    while True:
        bestobj = -np.inf
        bestcoln = None
        wherecoln = 'base'
        # iterate over possible columns to remove
        for coln in selected_base:
            tmp_sel_base = selected_base.copy()
            tmp_sel_base.remove(coln)
            fit = (fit_func(endog = resp, exog = X[tmp_sel_base], exog_infl = X[selected_infl])).fit(disp=0)
            beta = fit.params
            tmpobj = fit.llf - penalty(beta, len(resp))

            if (tmpobj > bestobj) and (tmpobj > curobj):
                print(len(beta))
                bestcoln = coln
                bestobj = tmpobj
                wherecoln = 'base' ## best coln is from base model

        for coln in selected_infl:
            tmp_sel_infl = selected_infl.copy()
            tmp_sel_infl.remove(coln)
            fit = (fit_func(endog = resp, exog = X[selected_base], exog_infl = X[tmp_sel_infl])).fit(disp=0)
            beta = fit.params
            tmpobj = fit.llf - penalty(beta, len(resp))

            if (tmpobj > bestobj) and (tmpobj > curobj):
                print(len(beta))
                bestcoln = coln
                bestobj = tmpobj
                wherecoln = 'infl'  ## base coln is from logistic model

        # if removing at least one column improved the objective, remove it and loop; otherwise quit
        if bestcoln is not None:
            if wherecoln == 'base':
                selected_base.remove(bestcoln)
            elif wherecoln == 'infl':
                selected_infl.remove(bestcoln)
            curobj = bestobj
            print(bestobj)
            print(wherecoln)
            print(selected_base + selected_infl)
        else:
            print('Done')
            break
    model = fit_func(endog=resp, exog=X[selected_base], exog_infl = X[selected_infl])
    res = model.fit(maxiter=100, disp=False)
    return curobj, selected_base, selected_infl, res

#fit zero-inflated neg bin

ZIBN_obj, ZIBN_selected_base, ZIBN_selected_infl, fit_ZIBN = forward_selection_ZIF(covariates, resp, bic, sm.ZeroInflatedNegativeBinomialP)

ZIBN_p_params = fit_ZIBN.params[0]
ZIBN_beta = fit_ZIBN.params[1:6]
ZIBN_alpha = fit_ZIBN.params.alpha

## compute residuals

pred_ZIBN = fit_ZIBN.predict()
pred_p = np.array(sp.special.expit(covariates[ZIBN_selected_infl]*ZIBN_p_params)).flatten()
sd_ZIBN_pred = ((1-pred_p)*(pred_ZIBN + np.power(pred_ZIBN, 2)*ZIBN_alpha + pred_p * np.power(pred_ZIBN,2)))
scale_ZIBN_res = (resp-pred_ZIBN)/(sd_ZIBN_pred)

## plot residuals

fig, ax = plt.subplots(figsize = (8,6))
ax.tick_params(labelsize=20)

ax.scatter(pred_ZIBN, scale_ZIBN_res)
ax.set_xlabel('Predicted Value', fontsize = 15)
ax.set_ylabel('Standardized Residuals', fontsize = 20)
ax.hlines(0, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1],
linestyles = '--', colors = 'r')
ax.set_title('ZIBN GLM', fontsize = 20)

#compute probabilities of perfect attendance

ZIBN_mean = np.exp(np.matmul(covariates[ZIBN_selected_base], ZIBN_beta))
ZIBN_pred_0_0 = pred_p[resp==0] + (1-pred_p[resp==0])*np.power(1 + ZIBN_alpha*ZIBN_mean[resp == 0], (-1/ZIBN_alpha))
print(ZIBN_pred_0_0.mean())
ZIBN_pred_0_all = pred_p + (1-pred_p)*np.power(1 + ZIBN_alpha*ZIBN_mean, (-1/ZIBN_alpha))
print(ZIBN_pred_0_all.mean())

##

##backwards with ZIP gives convergence warnings but still works
ZIP_obj, ZIP_selected_base, ZIP_selected_infl, fit_ZIP = backward_selection_ZIF(covariates, resp, bic, sm.ZeroInflatedPoisson)



## likelihood ratio test

dev = 2*(fit_ZIBN.llf - fit_negbin.llf)
csq2 = sp.stats.chi2.ppf(0.95, 2)
print(dev, csq2)

## bic objectives

print('pois bic obj' + str(obj_pois))
print('negbin bic obj' + str(obj_negbin))
print('zinb bic obj' + str(ZIBN_obj))

## print final zero inflated models

print(fit_ZIBN.summary())
print(fit_ZIP.summary())
